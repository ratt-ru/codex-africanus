#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
from collections import namedtuple
from functools import lru_cache
from operator import getitem
import weakref
import time

import numpy as np

try:
    from astropy.io import fits
    import dask
    import dask.array as da
    from dask.diagnostics import ProgressBar
    import Tigger
    from daskms import xds_from_ms, xds_from_table, xds_to_table
except ImportError as e:
    opt_import_error = e
else:
    opt_import_error = None

from africanus.util.beams import beam_filenames, beam_grids
from africanus.coordinates.dask import radec_to_lm
from africanus.rime.dask import (phase_delay, predict_vis, parallactic_angles,
                                 beam_cube_dde, feed_rotation, zernike_dde)
from africanus.model.coherency.dask import convert
from africanus.model.spectral.dask import spectral_model
from africanus.model.shape.dask import gaussian as gaussian_shape
from africanus.model.shape import gaussian as nb_gaussian
from africanus.model.shape.dask import shapelet as shapelet_fn
from africanus.model.shape import phase_steer_and_w_correct
from africanus.model.shape.dask import shapelet_with_w_term
from africanus.model.shape import shapelet as nb_shapelet
from africanus.util.requirements import requires_optional
from africanus.constants import c as lightspeed
import timeit
import time
from astropy.io import fits
import nifty_gridder as ng
import matplotlib.pyplot as plt


_einsum_corr_indices = 'ijkl'


def _brightness_schema(corrs, index):
    if corrs == 4:
        return "sf" + _einsum_corr_indices[index:index + 2], index + 1
    else:
        return "sfi", index


def _phase_delay_schema(corrs, index):
    return "srf", index


def _spi_schema(corrs, index):
    return "s", index


def _gauss_shape_schema(corrs, index):
    return "srf", index

def _shapelet_schema(corrs, index):
    return "rfs", index


def _bl_jones_output_schema(corrs, index):
    if corrs == 4:
        return "->srfi" + _einsum_corr_indices[index]
    else:
        return "->srfi"


_rime_term_map = {
    'brightness': _brightness_schema,
    'phase_delay': _phase_delay_schema,
    'spi': _spi_schema,
    'gauss_shape': _gauss_shape_schema,
    'shapelet_shape': _shapelet_schema,
}


def corr_schema(pol):
    """
    Parameters
    ----------
    pol : Dataset

    Returns
    -------
    corr_schema : list of list
        correlation schema from the POLARIZATION table,
        `[[9, 10], [11, 12]]` for example
    """

    # Select the single row out
    corrs = pol.NUM_CORR.data[0]
    corr_types = pol.CORR_TYPE.data[0]

    if corrs == 4:
        return [[corr_types[0], corr_types[1]],
                [corr_types[2], corr_types[3]]]  # (2, 2) shape
    elif corrs == 2:
        return [corr_types[0], corr_types[1]]    # (2, ) shape
    elif corrs == 1:
        return [corr_types[0]]                   # (1, ) shape
    else:
        raise ValueError("corrs %d not in (1, 2, 4)" % corrs)


def baseline_jones_multiply(corrs, *args):
    names = args[::2]
    arrays = args[1::2]

    input_einsum_schemas = []
    corr_index = 0

    for name, array in zip(names, arrays):
        try:
            # Obtain function for prescribing the input einsum schema
            schema_fn = _rime_term_map[name]
        except KeyError:
            raise ValueError("Unknown RIME term '%s'" % name)
        else:
            # Extract it and the next corr index
            einsum_schema, corr_index = schema_fn(corrs, corr_index)
            input_einsum_schemas.append(einsum_schema)

            if not len(einsum_schema) == array.ndim:
                raise ValueError("%s len(%s) == %d != %s.ndim"
                                 % (name, einsum_schema,
                                    len(einsum_schema), array.shape))

    output_schema = _bl_jones_output_schema(corrs, corr_index)
    schema = ",".join(input_einsum_schemas) + output_schema

    return da.einsum(schema, *arrays)


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms_one_hour")
    p.add_argument("ms_two_hours")
    p.add_argument("ms_three_hours")
    p.add_argument("-sm", "--sky-model", default="N6251-sky-model.txt")
    p.add_argument("-rc", "--row-chunks", type=int, default=10000)
    p.add_argument("-oc", "--output-coherence", action="store_true")
    p.add_argument("-mc", "--model-chunks", type=int, default=10)
    p.add_argument("-n", "--num-shapelet", type=int, default=30)
    p.add_argument("-b", "--beam", default=None)
    p.add_argument("-t", "--times", type=int, default=10)
    p.add_argument("-iuvw", "--invert-uvw", action="store_true",
                   help="Invert UVW coordinates. Useful if we want "
                        "compare our visibilities against MeqTrees")
    p.add_argument("-z", "--zernike", action="store_true")
    p.add_argument("-dc", "--data-column", type=str, default="MODEL_DATA")
    return p

def parse_sky_model(filename, chunks):
    """
    Parses a Tigger sky model

    Parameters
    ----------
    filename : str
        Sky Model filename
    chunks : tuple of ints or int
        Source chunking strategy

    Returns
    -------
    source_data : dict
        Dictionary of source data,
        :code:`{'point': (...), 'gauss': (...) }`
    """
    sky_model = Tigger.load(filename, verbose=False)

    _empty_spectrum = object()

    point_radec = []
    point_stokes = []
    point_spi = []
    point_ref_freq = []

    gauss_radec = []
    gauss_stokes = []
    gauss_spi = []
    gauss_ref_freq = []
    gauss_shape = []

    shapelet_radec = []
    shapelet_stokes = []
    shapelet_spi = []
    shapelet_ref_freq = []
    shapelet_beta = []
    shapelet_coeffs = []

    for source in sky_model.sources:
        ra = source.pos.ra
        dec = source.pos.dec
        typecode = source.typecode.lower()

        I = source.flux.I  # noqa
        Q = source.flux.Q
        U = source.flux.U
        V = source.flux.V

        spectrum = (getattr(source, "spectrum", _empty_spectrum)
                    or _empty_spectrum)

        try:
            # Extract reference frequency
            ref_freq = spectrum.freq0
        except AttributeError:
            ref_freq = sky_model.freq0

        try:
            # Extract SPI for I.
            # Zero Q, U and V to get 1 on the exponential
            spi = [[spectrum.spi, 0, 0, 0]]
        except AttributeError:
            # Default I SPI to -0.7
            spi = [[-0.7, 0, 0, 0]]

        if typecode == "gau":
            emaj = source.shape.ex
            emin = source.shape.ey
            pa = source.shape.pa

            gauss_radec.append([ra, dec])
            gauss_stokes.append([I, Q, U, V])
            gauss_spi.append(spi)
            gauss_ref_freq.append(ref_freq)
            gauss_shape.append([emaj, emin, pa])

        elif typecode == "pnt":
            point_radec.append([ra, dec])
            point_stokes.append([I, Q, U, V])
            point_spi.append(spi)
            point_ref_freq.append(ref_freq)

        elif typecode == "sha":
            beta_l = source.shape.sbetal
            beta_m = source.shape.sbetam
            coeffs = source.shape.shapelet_coeffs

            shapelet_radec.append([ra,dec])
            shapelet_stokes.append([I,Q,U,V])
            shapelet_spi.append(spi)
            shapelet_ref_freq.append(ref_freq)
            shapelet_beta.append([beta_l, beta_m])
            shapelet_coeffs.append(np.array(coeffs))
        else:
            raise ValueError("Unknown source morphology %s" % typecode)

    Point = namedtuple("Point", ["radec", "stokes", "spi", "ref_freq"])
    Gauss = namedtuple("Gauss", ["radec", "stokes", "spi", "ref_freq",
                                 "shape"])
    Shapelet = namedtuple("Shapelet", ["radec", "stokes", "spi", "ref_freq", "beta", "coeffs"])

    source_data = {}

    if len(point_radec) > 0:
        source_data['point'] = Point(
                    da.from_array(point_radec, chunks=(chunks, -1)),
                    da.from_array(point_stokes, chunks=(chunks, -1)),
                    da.from_array(point_spi, chunks=(chunks, 1, -1)),
                    da.from_array(point_ref_freq, chunks=chunks))
    if len(gauss_radec) > 0:
        source_data['gauss'] = Gauss(
                    da.from_array(gauss_radec, chunks=(chunks, -1)),
                    da.from_array(gauss_stokes, chunks=(chunks, -1)),
                    da.from_array(gauss_spi, chunks=(chunks, 1, -1)),
                    da.from_array(gauss_ref_freq, chunks=chunks),
                    da.from_array(gauss_shape, chunks=(chunks, -1)))
    if len(shapelet_radec) > 0:
        source_data['shapelet'] = Shapelet(
            da.from_array(shapelet_radec, chunks=(chunks, -1)),
            da.from_array(shapelet_stokes, chunks=(chunks, -1)),
            da.from_array(shapelet_spi, chunks=(chunks, 1, -1)),
            da.from_array(shapelet_ref_freq, chunks=(chunks)),
            da.from_array(shapelet_beta, chunks=(chunks, -1)),
            da.from_array(shapelet_coeffs, chunks=(chunks, 1, -1)))

    return source_data


def support_tables(args):
    """
    Parameters
    ----------
    args : object
        Script argument objects

    Returns
    -------
    table_map : dict of Dataset
        {name: dataset}
    """

    n = {k: '::'.join((args.ms_one_hour, k)) for k
         in ("ANTENNA", "DATA_DESCRIPTION", "FIELD",
             "SPECTRAL_WINDOW", "POLARIZATION")}

    # All rows at once
    lazy_tables = {"ANTENNA": xds_from_table(n["ANTENNA"])}

    compute_tables = {
        # Fixed shape rows
        "DATA_DESCRIPTION": xds_from_table(n["DATA_DESCRIPTION"]),
        # Variably shaped, need a dataset per row
        "FIELD": xds_from_table(n["FIELD"],
                                group_cols="__row__"),
        "SPECTRAL_WINDOW": xds_from_table(n["SPECTRAL_WINDOW"],
                                          group_cols="__row__"),
        "POLARIZATION": xds_from_table(n["POLARIZATION"],
                                       group_cols="__row__"),
    }

    lazy_tables.update(dask.compute(compute_tables)[0])
    return lazy_tables


def _zero_pes(parangles, frequency, dtype_):
    """ Create zeroed pointing errors """
    ntime, na = parangles.shape
    nchan = frequency.shape[0]
    return np.zeros((ntime, na, nchan, 2), dtype=dtype_)


def _unity_ant_scales(parangles, frequency, dtype_):
    """ Create zeroed antenna scalings """
    _, na = parangles[0].shape
    nchan = frequency.shape[0]
    return np.ones((na, nchan, 2), dtype=dtype_)

def findLargest(coeffs_array, n):
    # Find the n largest elements of shapelet coefficient array

    coeffs_copy = coeffs_array.copy()
    out_coeffs = np.zeros(coeffs_array.shape)
    for _ in range(n):
        highest_coeff = np.max(coeffs_copy)
        # print(np.where(coeffs_copy == highest_coeff)[:])
        # quit()
        highest_ind = np.where(coeffs_copy == highest_coeff)
        highest_ind = (highest_ind[0], highest_ind[1], highest_ind[2])
        out_coeffs[highest_ind] = highest_coeff
        coeffs_copy[highest_ind] = 0
    return out_coeffs

def my_timeit_shapelets(uvw, frequency, shapelet_coeffs , shapelet_beta, delta_lm, shapelet_lm, num_times):
    times = np.empty((num_times,))
    for i in range(num_times):
        t0 = time.time()
        s_fn = shapelet_with_w_term(uvw, frequency, shapelet_coeffs , shapelet_beta, delta_lm, shapelet_lm).compute()
        t1 = time.time()
        times[i] = t1 - t0
    return (np.sum(times) / times.shape[0]), s_fn

def my_timeit_nifty(uvw, frequency, dirty_image, weight, pixsize_x, pixsize_y, precision, nthreads, num_times):
    times = np.empty((num_times,))
    for i in range(num_times):
        t0 = time.time()
        ng_output = ng.dirty2ms(uvw, frequency, dirty_image, weight, pixsize_x, pixsize_y, precision, nthreads)
        t1 = time.time()
        times[i] = t1 - t0
    return (np.sum(times) / times.shape[0]), ng_output

def shapelet_factory(args, source_type, sky_model, ms, ant, field, spw, pol, ret_coherence=False, num_components=15):
    # Get data from measurement set
    phase_dir = field.PHASE_DIR.data[0][0]  # row, poly
    frequency = np.array(spw.CHAN_FREQ.data[0])
    uvw = (-ms.UVW.data if args.invert_uvw else ms.UVW.data).compute()

    # Create brightness matrix of 1 Jy
    stokes = np.array([[1,0,0,0]])
    brightness = convert(stokes, ["I", "Q", "U", "V"],
                        corr_schema(pol))

    # Get lm coordinates
    lm = radec_to_lm(sky_model.radec, phase_dir) # Same radec for both sources
    
    # Generate delta l and delta m
    delta_lm = np.array([1 / (10 * np.max(uvw[:, 0])), 1 / (10 * np.max(uvw[:, 1]))])

    # Generate visibilities from shapelets
    
    shapelet_beta = sky_model.beta.compute()
    shapelet_lm = lm.compute()
    no_components = num_components
    code_timings = np.empty((no_components,))
    uvw_chunks = (args.row_chunks, 3)
    frequency_chunks = (1,)
    beta_chunks = (1,2)
    delta_lm_chunks = (2,)
    shapelet_lm_chunks = (1,2)


    uvw = da.from_array(uvw, chunks=uvw_chunks)
    frequency = da.from_array(frequency, chunks=frequency_chunks)
    shapelet_beta = da.from_array(shapelet_beta, chunks=beta_chunks)
    delta_lm = da.from_array(delta_lm, chunks=delta_lm_chunks)
    shapelet_lm = da.from_array(shapelet_lm, chunks=shapelet_lm_chunks)


    for i in range(no_components):
        shapelet_coeffs = findLargest(sky_model.coeffs.compute(), i+1)
        shapelet_coeffs = da.from_array(shapelet_coeffs, chunks=shapelet_coeffs.shape)
        code_timings[i], s_fn = my_timeit_shapelets(uvw, frequency, shapelet_coeffs , shapelet_beta, delta_lm, shapelet_lm, args.times)
    
    if ret_coherence:
        shapelet_source_coherence = np.einsum("rsf,sij->srfij", s_fn, brightness.compute())
        shapelet_source_coherence = shapelet_source_coherence[0,:,0,:,:]
        return shapelet_source_coherence
    else:
        return s_fn[:,:,0] , code_timings



def nifty_factory(args, source_type, sky_model, ms, ant, field, spw, pol, fits_files, ret_coherence=False):
    # Get data from measurement set
    phase_dir = field.PHASE_DIR.data[0][0]  # row, poly
    phase_frequency = np.array(spw.CHAN_FREQ.data[0])
    uvw = (-ms.UVW.data if args.invert_uvw else ms.UVW.data).compute()

    # Create brightness matrix of 1 Jy
    stokes = np.array([[1,0,0,0]])
    brightness = convert(stokes, ["I", "Q", "U", "V"],
                        corr_schema(pol))

    # Get lm coordinates
    lm = radec_to_lm(sky_model.radec, phase_dir) # Same radec for both sources

    # Generate phase delay for Gaussian
    phase = phase_delay(lm, uvw, phase_frequency) 


    # Get measurement set data
    uvw = (-ms.UVW.data if args.invert_uvw else ms.UVW.data).compute()
    frequency = spw.CHAN_FREQ.data[0]

    # Set weight, pixel size, precision, and threads
    weight = None
    degree_pix_size = 0.02
    pixsize_x, pixsize_y = degree_pix_size * np.pi / 180 , degree_pix_size * np.pi / 180
    precision = 0.001
    nthreads = 16

    # Run nifty-gridder
    print("STARTING NIFTY NOW . . . .")
    code_timings = np.empty((args.num_shapelet,))
    for i in range(args.num_shapelet):
        # Gather data for nifty-gridder
        dirty_image = fits.open(fits_files[i])[0].data[0,0,:,:]
        code_timings[i], ng_output = my_timeit_nifty(uvw, frequency, dirty_image, weight, pixsize_x, pixsize_y, precision, nthreads, args.times)
    print("DONE WITH NIFTY")

    if ret_coherence:
        nifty_source_coherence = np.einsum("srf,rs,sij->srfij", phase.compute(), ng_output, brightness.compute())
        nifty_source_coherence = nifty_source_coherence[0,:,0,:,:]
        return nifty_source_coherence
    else:
        return ng_output, code_timings







    

@requires_optional("dask.array", "Tigger",
                   "daskms", opt_import_error)
def predict(args):
    # Convert source data into dask arrays
    sky_model = parse_sky_model(args.sky_model, args.model_chunks)

    # Get the support tables
    tables = support_tables(args)

    ant_ds = tables["ANTENNA"]
    field_ds = tables["FIELD"]
    ddid_ds = tables["DATA_DESCRIPTION"]
    spw_ds = tables["SPECTRAL_WINDOW"]
    pol_ds = tables["POLARIZATION"]

    code_timings = np.empty((args.num_shapelet, 3, 2))

    # Construct a graph for each DATA_DESC_ID
    for xds in xds_from_ms(args.ms_one_hour,
                        columns=["UVW", "ANTENNA1", "ANTENNA2", "TIME"],
                        group_cols=["FIELD_ID", "DATA_DESC_ID"],
                        chunks={"row": args.row_chunks}):

        # Perform subtable joins
        ant = ant_ds[0]
        field = field_ds[xds.attrs['FIELD_ID']]
        ddid = ddid_ds[xds.attrs['DATA_DESC_ID']]
        spw = spw_ds[ddid.SPECTRAL_WINDOW_ID.data[0]]
        pol = pol_ds[ddid.POLARIZATION_ID.data[0]]

        # Select single dataset row out
        corrs = pol.NUM_CORR.data[0]

        meta = np.empty((0,), dtype=tuple)
        utime_inv = xds.TIME.data.map_blocks(np.unique, return_inverse=True,
                                    meta=meta, dtype=tuple)
        time_idx = utime_inv.map_blocks(getitem, 1, dtype=np.int32)
        print(time_idx)

        print([stype for stype in sky_model.keys()])

        shapelet_result = shapelet_factory(args, 'shapelet', sky_model['shapelet'], xds, ant, field, spw, pol, ret_coherence=args.output_coherence, num_components=args.num_shapelet)
        nifty_result = nifty_factory(args, 'shapelet', sky_model['shapelet'], xds, ant, field, spw, pol,["shapelet_nifty_fits_files/shapelet-middle-res-%d-dirty.fits" %(n+1) for n in range(args.num_shapelet)], ret_coherence=args.output_coherence)

        if args.output_coherence:
            # Get correct coherence shape
            shapelet_coh = shapelet_result.reshape((1, shapelet_result.shape[0], 1) + (2,2))
            shapelet_coh = da.from_array(shapelet_coh)
            nifty_coh = nifty_result.reshape((1, nifty_result.shape[0],1) + (2,2))
            nifty_coh = da.from_array(nifty_coh)
            
            # Write visibilities for shapelets
            shapelet_vis = predict_vis(time_idx, xds.ANTENNA1.data, xds.ANTENNA2.data, None, shapelet_coh, None, None, None, None)
            nifty_vis = predict_vis(time_idx, xds.ANTENNA1.data, xds.ANTENNA2.data, None, nifty_coh, None, None, None, None)
            
            vis = shapelet_vis
            
            # Reshape (2, 2) correlation to shape (4,)
            if corrs == 4:
                vis = vis.reshape(vis.shape[:2] + (4,))
                nifty_vis = nifty_vis.reshape(nifty_vis.shape[:2] + (4,))

            # Assign visibilities to MODEL_DATA array on the dataset
            xds = xds.assign(MODEL_DATA=(("row", "chan", "corr"), vis))
            xds = xds.assign(CORRECTED_DATA=(("row", "chan", "corr"), nifty_vis))

            # Create a write to the table
            write = xds_to_table(xds, args.ms_one_hour, ["MODEL_DATA", "CORRECTED_DATA"])
            
            # Compute write
            with ProgressBar():
                da.compute(write)
        else:
            shapelet_result, shapelet_timings = shapelet_result
            nifty_result, nifty_timings = nifty_result
            code_timings[:,0,0], code_timings[:,0,1] = shapelet_timings, nifty_timings
            # plt.figure()
            # plt.plot(shapelet_timings)
            # plt.plot(nifty_timings)
            # plt.xlabel("Number of shapelet components")
            # plt.ylabel("Code timing (in seconds)")
            # plt.title("Shapelets vs Nifty Gridder")
            # plt.show()
            # plt.savefig("./nifty_timing_result.png")
            # plt.close()
            print(nifty_result.shape)
            print(shapelet_result.shape)
            nifty_norm = nifty_result / np.max(np.abs(nifty_result))
            shapelet_norm = shapelet_result / np.max(np.abs(shapelet_result))
            print("------------------------------------------------------")
            print("Shapelet and Gaussian Test (Normalized): ", np.allclose(shapelet_norm, nifty_norm))
            print("Phase Test: ", np.allclose(shapelet_norm.imag, nifty_norm.imag))
            print("Average difference (normalized, non-normalized):", np.average((shapelet_norm - nifty_norm)**2))
            print("Maximum difference (normalized, non-normalized):", np.max((shapelet_norm - nifty_norm)**2))
            print("------------------------------------------------------")
    if args.output_coherence:
        quit()
    for xds in xds_from_ms(args.ms_two_hours,
                        columns=["UVW", "ANTENNA1", "ANTENNA2", "TIME"],
                        group_cols=["FIELD_ID", "DATA_DESC_ID"],
                        chunks={"row": args.row_chunks}):

        # Perform subtable joins
        ant = ant_ds[0]
        field = field_ds[xds.attrs['FIELD_ID']]
        ddid = ddid_ds[xds.attrs['DATA_DESC_ID']]
        spw = spw_ds[ddid.SPECTRAL_WINDOW_ID.data[0]]
        pol = pol_ds[ddid.POLARIZATION_ID.data[0]]

        # Select single dataset row out
        corrs = pol.NUM_CORR.data[0]

        meta = np.empty((0,), dtype=tuple)
        utime_inv = xds.TIME.data.map_blocks(np.unique, return_inverse=True,
                                    meta=meta, dtype=tuple)
        time_idx = utime_inv.map_blocks(getitem, 1, dtype=np.int32)
        print(time_idx)

        print([stype for stype in sky_model.keys()])

        shapelet_result = shapelet_factory(args, 'shapelet', sky_model['shapelet'], xds, ant, field, spw, pol, ret_coherence=args.output_coherence, num_components=args.num_shapelet)
        nifty_result = None if args.output_coherence else nifty_factory(args, 'shapelet', sky_model['shapelet'], xds, ant, field, spw, pol,["shapelet_nifty_fits_files/shapelet-middle-res-%d-dirty.fits" %(n+1) for n in range(args.num_shapelet)], ret_coherence=args.output_coherence)

        if args.output_coherence:
            # Get correct coherence shape
            shapelet_coh = shapelet_result.reshape((1, shapelet_result.shape[0], 1) + (2,2))
            shapelet_coh = da.from_array(shapelet_coh)
            
            # Write visibilities for shapelets
            shapelet_vis = predict_vis(time_idx, xds.ANTENNA1.data, xds.ANTENNA2.data, None, shapelet_coh, None, None, None, None)
            
            vis = shapelet_vis
            
            # Reshape (2, 2) correlation to shape (4,)
            if corrs == 4:
                vis = vis.reshape(vis.shape[:2] + (4,))
               
            # Assign visibilities to MODEL_DATA array on the dataset
            xds = xds.assign(MODEL_DATA=(("row", "chan", "corr"), vis))
            
            # Create a write to the table
            write = xds_to_table(xds, args.ms_two_hours, ["MODEL_DATA"])
            
            # Compute write
            with ProgressBar():
                da.compute(write)
        else:
            shapelet_result, shapelet_timings = shapelet_result
            nifty_result, nifty_timings = nifty_result
            code_timings[:,1,0], code_timings[:,1,1] = shapelet_timings, nifty_timings
            # plt.figure()
            # plt.plot(shapelet_timings)
            # plt.plot(nifty_timings)
            # plt.xlabel("Number of shapelet components")
            # plt.ylabel("Code timing (in seconds)")
            # plt.title("Shapelets vs Nifty Gridder")
            # plt.show()
            # plt.savefig("./nifty_timing_result.png")
            # plt.close()
            print(nifty_result.shape)
            print(shapelet_result.shape)
            nifty_norm = nifty_result / np.max(np.abs(nifty_result))
            shapelet_norm = shapelet_result / np.max(np.abs(shapelet_result))
            print("------------------------------------------------------")
            print("Shapelet and Gaussian Test (Normalized): ", np.allclose(shapelet_norm, nifty_norm))
            print("Phase Test: ", np.allclose(shapelet_norm.imag, nifty_norm.imag))
            print("Average difference (normalized, non-normalized):", np.average((shapelet_norm - nifty_norm)**2))
            print("Maximum difference (normalized, non-normalized):", np.max((shapelet_norm - nifty_norm)**2))
            print("------------------------------------------------------")
    for xds in xds_from_ms(args.ms_three_hours,
                        columns=["UVW", "ANTENNA1", "ANTENNA2", "TIME"],
                        group_cols=["FIELD_ID", "DATA_DESC_ID"],
                        chunks={"row": args.row_chunks}):

        # Perform subtable joins
        ant = ant_ds[0]
        field = field_ds[xds.attrs['FIELD_ID']]
        ddid = ddid_ds[xds.attrs['DATA_DESC_ID']]
        spw = spw_ds[ddid.SPECTRAL_WINDOW_ID.data[0]]
        pol = pol_ds[ddid.POLARIZATION_ID.data[0]]

        # Select single dataset row out
        corrs = pol.NUM_CORR.data[0]

        meta = np.empty((0,), dtype=tuple)
        utime_inv = xds.TIME.data.map_blocks(np.unique, return_inverse=True,
                                    meta=meta, dtype=tuple)
        time_idx = utime_inv.map_blocks(getitem, 1, dtype=np.int32)
        print(time_idx)

        print([stype for stype in sky_model.keys()])

        shapelet_result = shapelet_factory(args, 'shapelet', sky_model['shapelet'], xds, ant, field, spw, pol, ret_coherence=args.output_coherence, num_components=args.num_shapelet)
        nifty_result = nifty_factory(args, 'shapelet', sky_model['shapelet'], xds, ant, field, spw, pol,["shapelet_nifty_fits_files/shapelet-middle-res-%d-dirty.fits" %(n+1) for n in range(args.num_shapelet)], ret_coherence=args.output_coherence)

        if args.output_coherence:
            # Get correct coherence shape
            shapelet_coh = shapelet_result.reshape((1, shapelet_result.shape[0], 1) + (2,2))
            shapelet_coh = da.from_array(shapelet_coh)
            nifty_coh = nifty_result.reshape((1, nifty_result.shape[0],1) + (2,2))
            nifty_coh = da.from_array(nifty_coh)
            
            # Write visibilities for shapelets
            shapelet_vis = predict_vis(time_idx, xds.ANTENNA1.data, xds.ANTENNA2.data, None, shapelet_coh, None, None, None, None)
            nifty_vis = predict_vis(time_idx, xds.ANTENNA1.data, xds.ANTENNA2.data, None, nifty_coh, None, None, None, None)
            
            vis = shapelet_vis
            
            # Reshape (2, 2) correlation to shape (4,)
            if corrs == 4:
                vis = vis.reshape(vis.shape[:2] + (4,))
                nifty_vis = nifty_vis.reshape(nifty_vis.shape[:2] + (4,))

            # Assign visibilities to MODEL_DATA array on the dataset
            xds = xds.assign(MODEL_DATA=(("row", "chan", "corr"), vis))
            xds = xds.assign(CORRECTED_DATA=(("row", "chan", "corr"), nifty_vis))

            # Create a write to the table
            write = xds_to_table(xds, args.ms_three_hours, ["MODEL_DATA", "CORRECTED_DATA"])
            # Compute write
            with ProgressBar():
                da.compute(write)
        else:
            shapelet_result, shapelet_timings = shapelet_result
            nifty_result, nifty_timings = nifty_result
            code_timings[:,2,0], code_timings[:,2,1] = shapelet_timings, nifty_timings
            # plt.figure()
            # plt.plot(shapelet_timings)
            # plt.plot(nifty_timings)
            # plt.xlabel("Number of shapelet components")
            # plt.ylabel("Code timing (in seconds)")
            # plt.title("Shapelets vs Nifty Gridder")
            # plt.show()
            # plt.savefig("./nifty_timing_result.png")
            # plt.close()
            print(nifty_result.shape)
            print(shapelet_result.shape)
            nifty_norm = nifty_result / np.max(np.abs(nifty_result))
            shapelet_norm = shapelet_result / np.max(np.abs(shapelet_result))
            print("------------------------------------------------------")
            print("Shapelet and Gaussian Test (Normalized): ", np.allclose(shapelet_norm, nifty_norm))
            print("Phase Test: ", np.allclose(shapelet_norm.imag, nifty_norm.imag))
            print("Average difference (normalized, non-normalized):", np.average((shapelet_norm - nifty_norm)**2))
            print("Maximum difference (normalized, non-normalized):", np.max((shapelet_norm - nifty_norm)**2))
            print("-----------------------------------------------------")
    plt.figure()
    # fig, ax = plt.subplots()
    plot_labels = [[""] * 2] * 3
    plot_labels[0][0] = "Shapelet (1 hour)"
    plot_labels[0][1] = "Nifty (1 hour)"
    plot_labels[1][0] = "Shapelet (2 hours)"
    plot_labels[1][1] = "Nifty (2 hours)"
    plot_labels[2][0] = "Shapelet (3 hours)"
    plot_labels[2][1] = "Nifty (3 hours)"
    for i in range(3):
        for j in range(2):
            print(code_timings[:,i,j])
            plt.plot(code_timings[:,i,j], label=plot_labels[i][j])
    plt.legend(loc='upper left', frameon=True)
    plt.xlabel("Number of shapelet components")
    plt.ylabel("Code timing (seconds)")
    plt.title("Shapelets vs Nifty Gridder")
    plt.savefig("./shapelet_nifty_timings_all.png")
    plt.close()


if __name__ == "__main__":
    args = create_parser().parse_args()
    predict(args)
