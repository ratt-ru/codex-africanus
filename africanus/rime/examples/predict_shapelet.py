#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from collections import namedtuple

from operator import getitem
import numpy as np

try:
    import dask
    import dask.array as da
    from dask.diagnostics import ProgressBar
    import Tigger
    import xarray as xr
    from xarrayms import xds_from_ms, xds_from_table, xds_to_table
except ImportError as e:
    opt_import_error = e
else:
    opt_import_error = None

from africanus.coordinates.dask import radec_to_lm
from africanus.rime.dask import phase_delay, predict_vis
from africanus.model.coherency.dask import convert
from africanus.model.spectral.dask import spectral_model
from africanus.model.shape.dask import gaussian as gaussian_shape
from africanus.model.shape.dask import shapelet as shapelet_fn
from africanus.rime.dask import zernike_dde
from africanus.rime.dask import parallactic_angles
from africanus.util.requirements import requires_optional
from astropy.io import fits
import time

# Testing stuff
from numpy.testing import assert_array_almost_equal
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


SOURCE_CHUNKS = 10
Fs = np.fft.fftshift

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



def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms")
    p.add_argument("-sm", "--sky-model", default="sky-model-shapelets.txt")
    p.add_argument("-rc", "--row-chunks", type=int, default=10000)
    p.add_argument("-mc", "--model-chunks", type=int, default=10000)
    p.add_argument("-iuvw", "--invert-uvw", action="store_true",
                   help="Invert UVW coordinates. Useful if we want "
                        "compare our visibilities against MeqTrees")
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
    radec : :class:`numpy.ndarray`
        :code:`(source, 2)` array of source coordinates
    stokes : :class:`numpy.ndarray`
        :code:`(source, 4)` array of stokes parameters
    """
    sky_model = Tigger.load(filename, verbose=False)
    

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
    max_shapelet_coeffs = 0


    for source in sky_model.sources:
        ra = source.pos.ra
        dec = source.pos.dec
        typecode = source.typecode.lower()

        I = source.flux.I
        Q = source.flux.Q
        U = source.flux.U
        V = source.flux.V

        try:
            ref_freq = source.freq0
        except AttributeError:
            ref_freq = sky_model.freq0

        try:
            spi = source.spi
        except AttributeError:
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
    print("radec",shapelet_radec)
    print("stokes",shapelet_stokes)
    print("spi",shapelet_spi)
    print("ref_freq",shapelet_ref_freq)
    print("beta",shapelet_beta)
    print("coeffs",shapelet_coeffs)
    
    return {
        'shapelet': Shapelet(da.from_array(shapelet_radec, chunks=(chunks)),
                             da.from_array(shapelet_stokes, chunks=(chunks, -1)),
                             da.from_array(shapelet_spi, chunks=(chunks, 1, -1)),
                             da.from_array(shapelet_ref_freq, chunks=chunks),
                             da.from_array(shapelet_beta, chunks=(chunks, -1)),
                             da.from_array(shapelet_coeffs, chunks=(chunks)))
    }


def support_tables(args, tables):
    """
    Parameters
    ----------
    args : object
        Script argument objects
    tables : list of str
        List of support tables to open

    Returns
    -------
    table_map : dict of :class:`xarray.Dataset`
        {name: dataset}
    """
    return {t: [ds.compute() for ds in
                xds_from_table("::".join((args.ms, t)),
                               group_cols="__row__")]
            for t in tables}


def corr_schema(pol):
    """
    Parameters
    ----------
    pol : :class:`xarray.Dataset`

    Returns
    -------
    corr_schema : list of list
        correlation schema from the POLARIZATION table,
        `[[9, 10], [11, 12]]` for example
    """

    corrs = pol.NUM_CORR.values
    corr_types = pol.CORR_TYPE.values

    if corrs == 4:
        return [[corr_types[0], corr_types[1]],
                [corr_types[2], corr_types[3]]]  # (2, 2) shape
    elif corrs == 2:
        return [corr_types[0], corr_types[1]]    # (2, ) shape
    elif corrs == 1:
        return [corr_types[0]]                   # (1, ) shape
    else:
        raise ValueError("corrs %d not in (1, 2, 4)" % corrs)


def write_fits(beam, timestamp, filename):
    """
    # Create header
    hdr = fits.Header()
    fMHz = np.array(freqs)*1e6
    if isinstance(fMHz, (int, float)): fMHz = [fMHz]
    try: df = fMHz[1]-fMHz[0]
    except: df = 1e6
    diam = float(diameter)
    if beam.shape[0]==2: xy = ['H', 'V']
    elif beam.shape[0]==4: xy = ['Mx', 'My']
    else: xy = ['', '']
    ctypes = ['px', 'py', 'FREQ', xy[0], xy[1]]
    crvals = [0.0, 0.0, fMHz[0], 0, 0, 0]
    cdelts = [diam/beam.shape[-2], diam/beam.shape[-1], df, 1, 1]
    cunits = ['deg', 'deg', 'Hz', '', '']
    nx, ny = beam.shape[-2], beam.shape[-1]
    if nx%2 == 0: crpixx, crpixy = nx/2, ny/2
    elif nx%2 == 1: crpixx, crpixy = int(nx/2), int(ny/2)
    crpixs = [crpixx, crpixy, 1, 1, 1]
    for i in range(len(beam.shape)):
        ii = str(i+1)
        hdr['CTYPE'+ii] = ctypes[i]
        hdr['CRPIX'+ii] = crpixs[i]
        hdr['CRVAL'+ii] = crvals[i]
        hdr['CDELT'+ii] = cdelts[i]
        hdr['CUNIT'+ii] = cunits[i]
    hdr['TELESCOP'] = 'MeerKAT'
    hdr['DATE'] = time.ctime()
    
    # Write real and imag parts of data
    hdu = fits.PrimaryHDU(beam, header=hdr)
    hdu.writeto(filename, overwrite=True)
    """

    hdr = fits.Header()
    ctypes = ['px', 'py']
    beam = beam.compute()
    crvals = [0.0, 0.0]
    crpix = [beam.shape[0] // 2, beam.shape[1] // 2]
    cunits = ["deg", "deg"]
    for i in range(len(beam.shape)):
        ii = str(i + 1)
        hdr['CTYPE' + ii] = ctypes[i]
        hdr['CRPIX' + ii] = crpix[i]
        print(crvals[i])
        hdr['CRVAL' + ii] = crvals[i]
        hdr['CUNIT' + ii] = cunits[i]
    hdr['TELESCOP'] = 'MeerKAT'
    hdr['DATE'] = time.ctime()
    print("COMPUTING BEAM")
    hdu = fits.PrimaryHDU(beam.real, header=hdr)
    print("HDU IS ", hdu.header)
    print("BEAM DONE")
    hdu.writeto(filename, overwrite=True)

def generate_primary_beam(filename, ant, chan, ntime, lm, pa, frequency_scaling, antenna_scaling, pointing_errors):
    npoly = 8
    coeffs_file = np.load(filename, allow_pickle=True, encoding="bytes").all()
    noll_indices = np.zeros((ant, chan, 2, 2, npoly))
    zernike_coeffs = np.zeros((ant, chan, 2,2,npoly), dtype=np.complex128)
    corr_letters = [b'x',b'y']
    nsrc = lm.shape[0]
    coords = np.empty((3, nsrc, ntime, ant, chan))
    lm = lm.compute()

    
    print("LM COORDINATES ARE ", lm)

    for a in range(ant):
        for c in range(chan):
            for t in range(ntime):
                coords[0, :, t, a, c] = lm[:,0]
                coords[1, :, t, a, c] = lm[:,1]
            for corr1 in range(2):
                for corr2 in range(2):
                    corr_index = corr_letters[corr1] + corr_letters[corr2]
                    noll_indices[a,c,corr1, corr2, :] = coeffs_file[b'noll_index'][corr_index][:npoly] 
                    zernike_coeffs[a,c,corr1,corr2, :] = coeffs_file[b'coeff'][corr_index][:npoly]
    z =  zernike_dde(da.from_array(coords, chunks=(3, 32, ntime, ant, chan)), 
            da.from_array(zernike_coeffs, chunks=zernike_coeffs.shape), 
            da.from_array(noll_indices, chunks=noll_indices.shape), 
            pa, 
            frequency_scaling, 
            antenna_scaling, 
            pointing_errors)

    """
    plt.figure("Zernike Beam")
    plt.imshow(np.abs(z.compute()[:, 0,0,0,0,0]).reshape((6,6)))
    plt.colorbar()
    plt.savefig("./partial_beam.png")
    plt.close()
    """
    return z

def generate_fov_primary_beam(filename, ant, chan, ntime, lm, pa, frequency_scaling, antenna_scaling, pointing_errors, npix):
    npoly = 8
    coeffs_file = np.load(filename, allow_pickle=True, encoding="bytes").all()
    noll_indices = np.zeros((ant, chan, 2, 2, npoly))
    zernike_coeffs = np.zeros((ant, chan, 2,2,npoly), dtype=np.complex128)
    corr_letters = [b'x',b'y']
    nsrc = lm.shape[0]
    coords = np.empty((3, nsrc, ntime, ant, chan))
    lm = lm.compute()

    
    print("LM COORDINATES ARE ", lm.shape, npix * npix)

    for a in range(ant):
        for c in range(chan):
            for t in range(ntime):
                coords[0, :, t, a, c] = lm[:,0]
                coords[1, :, t, a, c] = lm[:,1]
            for corr1 in range(2):
                for corr2 in range(2):
                    corr_index = corr_letters[corr1] + corr_letters[corr2]
                    noll_indices[a,c,corr1, corr2, :] = coeffs_file[b'noll_index'][corr_index][:npoly] 
                    zernike_coeffs[a,c,corr1,corr2, :] = coeffs_file[b'coeff'][corr_index][:npoly]
    print("CALLING ZERNIKE DDE")
    z =  zernike_dde(da.from_array(coords, chunks=(3, 32, ntime, ant, chan)), 
            da.from_array(zernike_coeffs, chunks=zernike_coeffs.shape), 
            da.from_array(noll_indices, chunks=noll_indices.shape), 
            pa, 
            frequency_scaling, 
            antenna_scaling, 
            pointing_errors)
    print("z is ", z[:, 0, 0, 0, 0, 0].shape)        
    z = z[:, 0, 0, 0, 0, 0].reshape((npix, npix))
    plt.figure()
    plt.imshow(z.real)
    plt.colorbar()
    plt.savefig("primary_beam.png")
    plt.close()
    print("ZERNIKE DDE DONE")

    """
    plt.figure("Zernike Beam")
    plt.imshow(np.abs(z.compute()[:, 0,0,0,0,0]).reshape((6,6)))
    plt.colorbar()
    plt.savefig("./partial_beam.png")
    plt.close()
    """
    write_fits(z, [0], "zernike_fits_beam.fits")

def baseline_jones_multiply(corrs, *args):
    names = args[::2]
    arrays = args[1::2]

    input_einsum_schemas = []
    corr_index = 0

    for name, array in zip(names, arrays):
        try:
            fn = _rime_term_map[name]
        except KeyError:
            raise ValueError("Unknown RIME term '%s'" % name)
        else:
            einsum_schema, corr_index = fn(corrs, corr_index)
            input_einsum_schemas.append(einsum_schema)

            if not len(einsum_schema) == array.ndim:
                raise ValueError("%s len(%s) == %d != %s.ndim"
                                 % (name, einsum_schema,
                                    len(einsum_schema), array.shape))
            assert len(einsum_schema) == array.ndim

    output_schema = _bl_jones_output_schema(corrs, corr_index)
    schema = ",".join(input_einsum_schemas) + output_schema

    return da.einsum(schema, *arrays)

def vis_factory(args, source_type, sky_model, time_index,
                ms, field, spw, pol, antenna_positions, utime):
    try:
        source = sky_model[source_type]
    except KeyError:
        raise ValueError("Source type '%s' unsupported" % source_type)

    corrs = pol.NUM_CORR.values

    lm = radec_to_lm(source.radec, field.PHASE_DIR.data)
    uvw = -ms.UVW.data if args.invert_uvw else ms.UVW.data
    frequency_vals = spw.CHAN_FREQ.data
    if frequency_vals.shape == ():
        frequency = np.empty((1,))
        frequency[0] == frequency_vals
    else:
        frequency = spw.CHAN_FREQ.data



    # (source, row, frequency)
    print(lm.compute() * 180 / np.pi)
    print(uvw)
    print(frequency.shape)
    phase = phase_delay(lm, uvw, frequency)

    # (source, spi, corrs)
    # Apply spectral mode to stokes parameters
    stokes = spectral_model(source.stokes,
                            source.spi,
                            source.ref_freq,
                            frequency,
                            base=0)

    brightness = convert(stokes, ["I", "Q", "U", "V"],
                         corr_schema(pol))



    args = ["phase_delay", phase]

    # Add any visibility amplitude terms
    if source_type == "gauss":
        args.append("gauss_shape")
        args.append(gaussian_shape(uvw, frequency, source.shape))
    if source_type == "shapelet":
        delta_lm = np.array([1 / (10 * np.max(uvw[:, 0])), 1 / (10 * np.max(uvw[:, 1]))])
        delta_lm = da.from_array(delta_lm, chunks=delta_lm.shape)
        frequency = da.from_array(frequency, chunks=frequency.size)
        args.append("shapelet_shape")
        args.append(shapelet_fn(uvw, frequency, source.coeffs, source.beta, delta_lm))
        print(shapelet_fn(uvw, frequency, source.coeffs, source.beta, delta_lm))

    args.extend(["brightness", brightness])

    jones = baseline_jones_multiply(corrs, *args)
    

    ntime = len(utime)
    nchan = len(frequency)
    na = len(antenna_positions)
    delta_lm = np.array([1 / (10 * np.max(uvw[:, 0])), 1 / (10 * np.max(uvw[:, 1]))])

    generate_zernikes = False
    if generate_zernikes:
        print("GENERATING ZERNIKE PRIMARY BEAM")

        # Create frequency_scaling and antenna_scaling for primary beam
        frequency_scaling = np.ones((nchan,), dtype=np.float64)
        antenna_scaling = np.ones((na, nchan, 2), dtype=np.float64)
        pointing_errors = np.zeros((ntime, na, nchan, 2), dtype=np.float64)

        # Compute parallactic_angle
        pa = parallactic_angles(utime, da.from_array(antenna_positions), da.from_array(field.PHASE_DIR.data)) 

        write_fits_primary_beam = True
        if write_fits_primary_beam:
            fov = 2.0

            npix = 65

            nx, ny = npix, npix
            grid = (np.indices((nx, ny), dtype=np.float) - nx//2) * fov / nx
            ll, mm = grid[0], grid[1]

            lm = np.vstack((ll.flatten(), mm.flatten())).T

            print("lm center is ", lm[0,:], lm[-1, :])

            coords = np.empty((npix ** 2, 3),dtype=np.float64)

            coords[:, :2], coords[:, 2] = lm, 0

            coords = da.from_array(coords, chunks=coords.shape)



            print("lm grid shape: ", lm.shape)

            generate_fov_primary_beam("./zernike_coeffs.npy", na, nchan, ntime, coords, pa, frequency_scaling, antenna_scaling, pointing_errors, npix)

            quit()

        # Create primary beam
        dde_primary_beam = generate_primary_beam("./zernike_coeffs.npy", na, nchan, ntime, lm, pa, frequency_scaling, antenna_scaling, pointing_errors) * -1

        # return predict_vis(time_index, ms.ANTENNA1.data, ms.ANTENNA2.data,
        #                dde_primary_beam, jones, dde_primary_beam, None, None, None)
        
        return predict_vis(time_index, ms.ANTENNA1.data, ms.ANTENNA2.data,
                       dde_primary_beam, jones, dde_primary_beam, np.ones((ntime, na, nchan, 2, 2), dtype=np.float64), None, np.ones((ntime, na, nchan, 2, 2), dtype=np.float64))
    else:
        print("SKIPPING PRIMARY BEAM")
        return predict_vis(time_index, ms.ANTENNA1.data, ms.ANTENNA2.data,
                       None, jones, None, None, None, None)
      


@requires_optional("dask.array", "Tigger",
                   "xarray", "xarrayms", opt_import_error)
def predict(args):
    # Convert source data into dask arrays
    sky_model = parse_sky_model(args.sky_model, args.model_chunks)

    # Get the support tables
    tables = support_tables(args, ["FIELD", "DATA_DESCRIPTION",
                                   "SPECTRAL_WINDOW", "POLARIZATION", "ANTENNA"])

    field_ds = tables["FIELD"]
    ddid_ds = tables["DATA_DESCRIPTION"]
    spw_ds = tables["SPECTRAL_WINDOW"]
    pol_ds = tables["POLARIZATION"] 
    ant_ds = tables["ANTENNA"]
    antenna_positions = []
    for a in range(len(ant_ds)):
        antenna_positions.append(ant_ds[a].POSITION.data)

    # List of write operations
    writes = []

    # Construct a graph for each DATA_DESC_ID
    for xds in xds_from_ms(args.ms,
                           columns=["UVW", "ANTENNA1", "ANTENNA2", "TIME"],
                           group_cols=["FIELD_ID", "DATA_DESC_ID"],
                           chunks={"row": args.row_chunks}):

        # Extract frequencies from the spectral window associated
        # with this data descriptor id
        field = field_ds[xds.attrs['FIELD_ID']]
        ddid = ddid_ds[xds.attrs['DATA_DESC_ID']]
        spw = spw_ds[ddid.SPECTRAL_WINDOW_ID.values]
        pol = pol_ds[ddid.POLARIZATION_ID.values]
        frequency = spw.CHAN_FREQ.data

        corrs = pol.NUM_CORR.values
        time = xds.TIME.data
        intermediate = time.map_blocks(lambda t: np.unique(t, return_inverse=True),
                                        meta=np.empty((0,), dtype=tuple),
                                        dtype=tuple)
        utime = intermediate.map_blocks(lambda i: getitem(i, 0),
                                        chunks=(np.nan,),
                                        dtype=time.dtype)
        time_index = intermediate.map_blocks(lambda i: getitem(i, 1),
                                        dtype=np.int32)
                                    
        utime = da.from_array(utime.compute())

        # Generate visibility expressions for each source type
        source_vis = [vis_factory(args, stype, sky_model, time_index,
                                  xds, field, spw, pol, antenna_positions, utime)
                      for stype in sky_model.keys()]


        # Sum visibilities together
        vis = sum(source_vis)

        # Reshape (2, 2) correlation to shape (4,)
        if corrs == 4:
            vis = vis.reshape(vis.shape[:2] + (4,))

        # Assign visibilities to MODEL_DATA array on the dataset
        model_data = xr.DataArray(vis, dims=["row", "chan", "corr"])
        xds = xds.assign(MODEL_DATA=model_data)
        # Create a write to the table
        write = xds_to_table(xds, args.ms, ['MODEL_DATA'])
        # Add to the list of writes
        writes.append(write)

    # Submit all graph computations in parallel
    with ProgressBar():
        dask.compute(writes)

if __name__ == "__main__":
    args = create_parser().parse_args()
    predict(args)