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
from africanus.model.shape import shapelet_with_w_term
from africanus.model.shape import shapelet as nb_shapelet
from africanus.util.requirements import requires_optional
from africanus.constants import c as lightspeed
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
    p.add_argument("ms")
    p.add_argument("-sm1", "--sky-model-1", default="shapelet_zero_order.txt")
    p.add_argument("-sm2", "--sky-model-2", default="gaussian-zero-order.txt")
    p.add_argument("-rc", "--row-chunks", type=int, default=10000)
    p.add_argument("-vs", "--vis-shapelet", action="store_true")
    p.add_argument("-vg", "--vis-gauss", action="store_true")
    p.add_argument("-mc", "--model-chunks", type=int, default=10)
    p.add_argument("-b", "--beam", default=None)
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

    n = {k: '::'.join((args.ms, k)) for k
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



def coherence_factory(args, source_type1, sky_model1, source_type2, sky_model2, ms, ant, field, spw, pol, ret_shapelet=False, ret_gauss=False):
    # Get data from measurement set
    phase_dir = field.PHASE_DIR.data[0][0]  # row, poly
    phase_frequency = np.array(spw.CHAN_FREQ.data[0])
    f = phase_frequency[0]
    uvw = (-ms.UVW.data if args.invert_uvw else ms.UVW.data).compute()
    # print(phase_frequency)
    # quit()

    # Get data from sky models
    source1 = sky_model1[source_type1]
    source2 = sky_model2[source_type2]

    # Create brightness matrix of 1 Jy
    stokes = np.array([[1,0,0,0]])
    brightness = convert(stokes, ["I", "Q", "U", "V"],
                        corr_schema(pol))

    # Get lm coordinates
    lm = radec_to_lm(source1.radec, phase_dir) # Same radec for both sources

    # Generate phase delay for Gaussian
    phase = phase_delay(lm, uvw, phase_frequency) 
    
    # Create Gaussian shape function
    gaussian_null_frequency = (2.0 * lightspeed * np.sqrt(np.log(2.0))) / (np.pi * f)
    gaussian_shape_function = nb_gaussian(uvw, phase_frequency, (source2.shape.compute() * gaussian_null_frequency)) # We multiply with a nullifying frequency parameter to cancel the effects of frequency scaling
    
    # Generate Gaussian source coherency
    gaussian_source_coherence = np.einsum("srf,srf,sij->srfij", phase.compute(), gaussian_shape_function, brightness.compute())
    gaussian_source_coherence = gaussian_source_coherence[0,:,0,:,:] # Only interested in the row and correlation dimensions
    

    # Write shapelet_with_w_term here
    delta_lm = np.array([1 / (10 * np.max(uvw[:, 0])), 1 / (10 * np.max(uvw[:, 1]))])
    shapelet_null_frequency =  (lightspeed / (2 * np.pi * f))
    shapelet_beta = source1.beta.compute()
    shapelet_beta *= shapelet_null_frequency # To nullify frequency scaling to match Gaussian
    shapelet_beta *= 1 / (2.0 * np.pi) # To correct for the xx * 2 * np.pi

    # Get Shapelet Source Coherence with W-Term
    shapelet_shape_function = shapelet_with_w_term(uvw, phase_frequency, source1.coeffs.compute(), shapelet_beta, delta_lm, lm.compute()) # shape (nrow, 1, 1)
    shapelet_source_coherence = np.einsum("rsf,sij->srfij", shapelet_shape_function, brightness.compute())
    shapelet_source_coherence = shapelet_source_coherence[0,:,0,:,:]

    # Test w term
    w_term = np.empty((uvw.shape[0],), dtype=np.complex128)
    lm = lm.compute()[0,:]
    for r in range(uvw.shape[0]):
        freq_coords = uvw[r,:]
        w_term[r] = phase_steer_and_w_correct(freq_coords, lm, phase_frequency)
    print("W-terms match: ",np.allclose(phase.compute()[0,:,0], w_term))
    # quit()

    
    # Get Shapelet Source Coherence without W-term
    # shapelet_shape_function = nb_shapelet(uvw, shapelet_null_frequency, source1.coeffs.compute(), shapelet_beta, delta_lm)
    # shapelet_source_coherence = np.einsum("srf,rsf,sij->srfij", phase.compute(), shapelet_shape_function, brightness.compute())
    # shapelet_source_coherence = shapelet_source_coherence[0,:,0,:,:]



    if ret_shapelet or ret_gauss:
        return (shapelet_source_coherence if ret_shapelet else None, gaussian_source_coherence if ret_gauss else None)
    else:
        gauss_norm = gaussian_source_coherence / np.max(np.abs(gaussian_source_coherence))
        shapelet_norm = shapelet_source_coherence / np.max(np.abs(shapelet_source_coherence))
        print("------------------------------------------------------")
        print("Shapelet and Gaussian Test (Normalized): ", np.allclose(shapelet_norm, gauss_norm))
        print("Phase Test: ", np.allclose(shapelet_norm.imag, gauss_norm.imag))
        print("Shapelet and Gaussian Test: ", np.allclose(shapelet_source_coherence, gaussian_source_coherence))
        print("Average difference (normalized, non-normalized):", np.average((shapelet_norm - gauss_norm)**2), np.average((shapelet_source_coherence - gaussian_source_coherence)**2))
        print("Maximum difference (normalized, non-normalized):", np.max((shapelet_norm - gauss_norm)**2), np.max((shapelet_source_coherence - gaussian_source_coherence)**2))
        print("------------------------------------------------------")
        quit()


    

@requires_optional("dask.array", "Tigger",
                   "daskms", opt_import_error)
def predict(args):
    # Convert source data into dask arrays
    sky_model_1 = parse_sky_model(args.sky_model_1, args.model_chunks)

    sky_model_2 = parse_sky_model(args.sky_model_2, args.model_chunks)

    # Get the support tables
    tables = support_tables(args)

    ant_ds = tables["ANTENNA"]
    field_ds = tables["FIELD"]
    ddid_ds = tables["DATA_DESCRIPTION"]
    spw_ds = tables["SPECTRAL_WINDOW"]
    pol_ds = tables["POLARIZATION"]

    # List of write operations
    writes = []

    # Construct a graph for each DATA_DESC_ID
    sm_visibilities = [None, None]
    i = 0
    for sky_model in [sky_model_1, sky_model_2]:
        for xds in xds_from_ms(args.ms,
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

            print([stype for stype in sky_model_2.keys()])
            ret_shapelet = args.vis_shapelet
            ret_gauss = args.vis_gauss

            shapelet_coh, gauss_coh = coherence_factory(args, 'shapelet', sky_model_1, 'gauss', sky_model_2, xds, ant, field, spw, pol, ret_shapelet=ret_shapelet, ret_gauss=ret_gauss)
            if ret_shapelet:
                # Get correct coherence shape
                shapelet_coh = shapelet_coh.reshape((1, shapelet_coh.shape[0], 1) + (2,2))
                shapelet_coh = da.from_array(shapelet_coh)
                
                # Write visibilities for shapelets
                shapelet_vis = predict_vis(time_idx, xds.ANTENNA1.data, xds.ANTENNA2.data, None, shapelet_coh, None, None, None, None)
              
                # Generate visibility expressions for each source type
                # source_vis = [vis_factory(args, stype, sky_model,
                #                         xds, ant, field, spw, pol)
                #             for stype in sky_model.keys()]

                vis = shapelet_vis#sum(source_vis)
                
                # Reshape (2, 2) correlation to shape (4,)
                if corrs == 4:
                    vis = vis.reshape(vis.shape[:2] + (4,))

                # Assign visibilities to MODEL_DATA array on the dataset
                xds = xds.assign(MODEL_DATA=(("row", "chan", "corr"), vis))

                # Create a write to the table
                write = xds_to_table(xds, args.ms, ["MODEL_DATA"])
                print("WRITING SHAPELETS TO MODEL_DATA")

                # Compute write
                with ProgressBar():
                    da.compute(write)
            if ret_gauss:
                # Get correct coherence shape
                gauss_coh = gauss_coh.reshape((1, gauss_coh.shape[0], 1) + (2,2))
                gauss_coh = da.from_array(gauss_coh)
                
                # Write visibilities for Gaussians
                gauss_vis = predict_vis(time_idx, xds.ANTENNA1.data, xds.ANTENNA2.data, None, gauss_coh, None, None, None, None)
              
                # Generate visibility expressions for each source type
                # source_vis = [vis_factory(args, stype, sky_model,
                #                         xds, ant, field, spw, pol)
                #             for stype in sky_model.keys()]

                vis = gauss_vis#sum(source_vis)
                
                # Reshape (2, 2) correlation to shape (4,)
                if corrs == 4:
                    vis = vis.reshape(vis.shape[:2] + (4,))

                # Assign visibilities to MODEL_DATA array on the dataset
                xds = xds.assign(CORRECTED_DATA=(("row", "chan", "corr"), vis))

                # Create a write to the table
                write = xds_to_table(xds, args.ms, ["CORRECTED_DATA"])
                print("WRITING GAUSSIAN TO CORRECTED_DATA")

                # Compute write
                with ProgressBar():
                    da.compute(write)
            quit()

            # Add to the list of writes
            writes.append(write)


    # Submit all graph computations in parallel
    with ProgressBar():
        dask.compute(writes)


if __name__ == "__main__":
    args = create_parser().parse_args()
    predict(args)
