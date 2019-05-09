#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from dask.diagnostics import ProgressBar
import numpy as np


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


SOURCE_CHUNKS = 10


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms")
    p.add_argument("-sm", "--sky-model", default="sky-model.txt")
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

    data = np.loadtxt(filename, delimiter=",", ndmin=1,
                      converters=converters, dtype=dtype)

    # Transpose
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

        # (source, row, frequency)
        phase = phase_delay(lm, uvw, frequency)

        brightness = convert(stokes, ["I", "Q", "U", "V"],
                             corr_schema(pol))

        # (source, row, frequency, corr1, corr2)
        jones = da.einsum(einsum_schema(pol), phase, brightness)

        # Identify time indices
        _, time_index = da.unique(xds.TIME.data, return_inverse=True)

        # Predict visibilities
        vis = predict_vis(time_index, xds.ANTENNA1.data, xds.ANTENNA2.data,
                          None, jones, None, None, None, None)

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
