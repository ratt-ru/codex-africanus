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
from africanus.compatibility import string_types
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
    converters = {
        0: lambda c: Angle(c).rad, 1: lambda c: Angle(c).rad,
        2: float, 3: float, 4: float, 5: float}

    dtype = {
        'names': ("ra", "dec", "I", "Q", "U", "V"),
        'formats': (np.float64,)*6}

    data = np.loadtxt(filename, delimiter=",",
                      converters=converters, dtype=dtype)

    # Transpose
    data = zip(*data)

    # Convert to numpy arrays
    ra, dec, i, q, u, v = (np.asarray(a) for a in data)
    radec = np.stack([ra, dec], axis=1)
    stokes = np.stack([i, q, u, v], axis=1)

    return radec, stokes


def support_tables(args, tables):
    return {t: [ds.compute() for ds in
                xds_from_table("::".join((args.ms, t)), group_cols="__row__")]
            for t in tables}


def corr_and_einsum_schema(pol):
    corrs = pol.NUM_CORR.values
    corr_types = pol.CORR_TYPE.values

    if corrs == 4:
        corr_schema = [[corr_types[0], corr_types[1]],
                       [corr_types[2], corr_types[3]]]
        einsum_schema = "srf, sij -> srfij"
    elif corrs == 2:
        corr_schema = [corr_type[0], corr_type[1]]
        einsum_schema = "srf, si -> srfi"
    elif corrs == 1:
        corr_schema = [corr_type[0]]
        einsum_schema = "srf, si -> srfi"
    else:
        raise ValueError("corrs %d not in (1, 2, 4)" % corrs)

    return corr_schema, einsum_schema


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

        corr_schema, einsum_schema = corr_and_einsum_schema(pol)

        brightness = convert(stokes, ["I", "Q", "U", "V"],
                             corr_schema)

        # (source, row, frequency, corr1, corr2)
        jones = da.einsum(einsum_schema, phase, brightness)

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
