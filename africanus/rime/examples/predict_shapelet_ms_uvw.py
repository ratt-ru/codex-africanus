#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from dask.diagnostics import ProgressBar
import numpy as np
import shapelets as sl

try:
    from astropy.coordinates import Angle
except ImportError as e:
    astropy_import_error = e
else:
    astropy_import_error = None

try:
    import dask
    import dask.array as da
    import xarray as xr
    from xarrayms import xds_from_ms, xds_from_table, xds_to_table
except ImportError as e:
    opt_import_error = e
else:
    opt_import_error = None

from africanus.coordinates.dask import radec_to_lm
from africanus.rime.dask import phase_delay, predict_vis
from africanus.model.coherency.dask import convert
from africanus.util.requirements import requires_optional
from africanus.model.shape.dask import shapelet as shapelet_fn
from numpy.testing import assert_array_almost_equal
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


SOURCE_CHUNKS = 10
Fs = np.fft.fftshift

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms")
    p.add_argument("-sm", "--sky-model", default="sky-model-shapelets.txt")
    p.add_argument("-rc", "--row-chunks", type=int, default=10000)
    p.add_argument("-iuvw", "--invert-uvw", action="store_true",
                   help="Invert UVW coordinates. Useful if we want "
                        "compare our visibilities against MeqTrees")
    return p


@requires_optional('astropy', astropy_import_error)
def parse_sky_model(filename):
    """
    Parameters
    ----------
    filename : str
        Sky Model filename

    Returns
    -------
    radec : :class:`numpy.ndarray`
        :code:`(source, 2)` array of source coordinates
    stokes : :class:`numpy.ndarray`
        :code:`(source, 4)` array of stokes parameters
    """
    converters = {
        0: lambda c: Angle(c).rad, 1: lambda c: Angle(c).rad,
        2: float, 3: float, 4: float, 5: float}

    dtype = {
        'names': ("ra", "dec", "I", "Q", "U", "V"),
        'formats': (np.float64,)*6}

    data = np.loadtxt(filename, delimiter=",",
                      converters=converters, dtype=dtype, ndmin=1)

    # Should have lists of values for each row.
    # raw numpy array implies only a single row, convert to list
    data = zip(*data)


    # Convert to numpy arrays
    ra, dec, i, q, u, v = (np.asarray(a) for a in data)
    radec = np.stack([ra, dec], axis=1)
    stokes = np.stack([i, q, u, v], axis=1)

    return radec, stokes


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


def einsum_schema(pol):
    """
    Returns an einsum schema suitable for multiplying per-baseline
    phase and brightness terms.

    Parameters
    ----------
    pol : :class:`xarray.Dataset`

    Returns
    -------
    einsum_schema : str
    """
    corrs = pol.NUM_CORR.values

    if corrs == 4:
        return "srf, sij -> srfij"
    elif corrs in (2, 1):
        return "srf, si -> srfi"
    else:
        raise ValueError("corrs %d not in (1, 2, 4)" % corrs)

def _verify_shapelets(shapelets, uv, beta_vals, coeffs):
    nrow = shapelets.shape[1]
    gf_shapelets = np.zeros((nrow), dtype=np.complex128)
    nmax = [coeffs.shape[1], coeffs.shape[2]]
    for n1 in range(nmax[0]):
        for n2 in range(nmax[1]):
            gf_dimbasis = sl.shapelet.dimBasis2d(n1, n2, beta=beta_vals, fourier=True)
            gf_coeffs = coeffs[0, n1, n2]
            gf_shapelets += gf_coeffs * sl.shapelet.computeBasis2d(gf_dimbasis, uv[:, 0], uv[:, 1])
    assert_array_almost_equal(shapelets, gf_shapelets.reshape(shapelets.shape))

@requires_optional("dask.array", "xarray", "xarrayms", opt_import_error)
def predict(args):
    # Numpy arrays

    # Convert source data into dask arrays
    radec, stokes = parse_sky_model(args.sky_model)
    radec = da.from_array(radec, chunks=(SOURCE_CHUNKS, 2))
    stokes = da.from_array(stokes, chunks=(SOURCE_CHUNKS, 4))

    # Get the support tables
    tables = support_tables(args, ["FIELD", "DATA_DESCRIPTION",
                                   "SPECTRAL_WINDOW", "POLARIZATION"])

    field_ds = tables["FIELD"]
    ddid_ds = tables["DATA_DESCRIPTION"]
    spw_ds = tables["SPECTRAL_WINDOW"]
    pol_ds = tables["POLARIZATION"]

    shapelet_beta = np.array([[1., 1.]])
    shapelet_coeffs = np.array([[[1.]]])

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

        lm = radec_to_lm(radec, field.PHASE_DIR.data)
        uvw = -xds.UVW.data if args.invert_uvw else xds.UVW.data
        """
        nu = 90
        nv = 84

        u_range = [-3 * np.sqrt(2) *(shapelet_beta[0, 0] ** (-1)), 3 * np.sqrt(2) * (shapelet_beta[0, 0] ** (-1))]
        v_range = [-3 * np.sqrt(2) *(shapelet_beta[0, 1] ** (-1)), 3 * np.sqrt(2) * (shapelet_beta[0, 1] ** (-1))]


        du = (u_range[1] - u_range[0]) / nu
        dv = (v_range[1] - v_range[0]) / nv
        freqs_u = Fs(np.fft.fftfreq(nu, d=du))
        freqs_v = Fs(np.fft.fftfreq(nv, d=dv))
        uu, vv = np.meshgrid(freqs_u, freqs_v)
        uv = np.vstack((uu.flatten(), vv.flatten())).T

        uvw = np.empty((90 * 84, 3), dtype=np.float)

        uvw[:, :2], uvw[:, 2] = uv, 0



        uvw = da.from_array(uvw, chunks=uvw.shape)
        """
        # (source, row, frequency)
        phase = phase_delay(lm, uvw, frequency)

        shapelets = shapelet_fn(da.from_array(uvw, chunks=uvw.shape), da.from_array(shapelet_coeffs, chunks=shapelet_coeffs.shape), da.from_array(shapelet_beta, chunks=shapelet_beta.shape))
        _verify_shapelets(shapelets, uvw.compute(), shapelet_beta[0], shapelet_coeffs)
        
        plt.figure()
        plt.imshow(np.abs(shapelets.compute()[0, :].reshape((90, 84))))
        plt.colorbar()
        plt.title("Shapelets_MS_UVW")
       # plt.show()
        plt.savefig("shapelets_ms_uvw.png")
        plt.close()
        
        brightness = convert(stokes, ["I", "Q", "U", "V"],
                             corr_schema(pol))
#use pywrap in python
#casalite casa browser
        # (source, row, frequency, corr1, corr2)

        #jones = da.einsum(einsum_schema(pol), shapelets, phase, brightness)
        phase_shapelet_einsum = da.einsum("sr, srf -> srf", shapelets, phase)
        print(phase_shapelet_einsum)
        print(phase)
        """
        plt.figure()
        plt.imshow(np.abs(phase_shapelet_einsum.compute()[0, :, 0].reshape((90, 84))))
        plt.colorbar()
        plt.title("phase_shapelet_einsum_MS_UVW")
        #plt.show()
        plt.savefig("phase_shapelet_einsum_ms_uvw.png")
        plt.close()
        print("Created phase_shapelet_einsum image")
        """
        jones = da.einsum(einsum_schema(pol), phase_shapelet_einsum, brightness)
        print(jones)
        """
        plt.figure()
        plt.imshow(np.abs(jones.compute()[0, :, 0, 0, 0].reshape((90, 84))))
        plt.title("Phase_MS_UVW")
        plt.colorbar()
        #plt.show()
        plt.savefig("jones_ms_uvw.png")
        plt.close()
        print("Created Jones image")
        """
        # Identify time indices
        _, time_index = da.unique(xds.TIME.data, return_inverse=True)

        # Predict visibilities
        vis = predict_vis(time_index, xds.ANTENNA1.data, xds.ANTENNA2.data,
                          None, jones, None, None, None, None)

        # Reshape (2, 2) correlation to shape (4,)
        if corrs == 4:
            vis = vis.reshape(vis.shape[:2] + (4,))

        model_shapelets = np.empty(vis.shape, dtype=np.complex128)
        for chan in range(vis.shape[1]):
            for corr in range(corrs):
                model_shapelets[:, chan, corr] = shapelets.compute()[0, :]
        model_shapelets = da.from_array(model_shapelets, chunks=model_shapelets.shape)
        # Assign visibilities to MODEL_DATA array on the dataset
        model_data = xr.DataArray(vis, dims=["row", "chan", "corr"])
        #model_data = xr.DataArray(model_shapelets, dims=["row", "chan", "corr"])
        xds = xds.assign(MODEL_DATA=model_data)
        # Create a write to the table
        write = xds_to_table(xds, args.ms, ['MODEL_DATA'])
        # Add to the list of writes
        writes.append(write)
        print(model_data)
        md = model_data[:, 0, 0].data.reshape((90, 84))
        """
        plt.figure()
        plt.imshow(np.abs(md))
        plt.colorbar()
        plt.title("Predict_Image_MS_UVW")
        plt.savefig("./predict_img_ms_uvw.png")
        plt.close()
        print("Created predict image")
        """
        
    # Submit all graph computations in parallel
    with ProgressBar():
        dask.compute(writes)


if __name__ == "__main__":
    args = create_parser().parse_args()
    predict(args)
