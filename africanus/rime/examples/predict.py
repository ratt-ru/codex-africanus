#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np

try:
    import dask
    import dask.array as da
except ImportError as e:
    da_import_error = e
else:
    da_import_error = None

try:
    import xarray as xr
except ImportError as e:
    xr_import_error = e
else:
    xr_import_error = None

try:
    from xarrayms import xds_from_ms, xds_from_table, xds_to_table
except ImportError as e:
    xms_import_error = e
else:
    xms_import_error = None

from africanus.coordinates.dask import radec_to_lm
from africanus.rime.dask import phase_delay, predict_vis
from africanus.stokes.dask import stokes_convert
from africanus.util.requirements import requires_optional


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms")
    p.add_argument("-rc", "--row-chunks", type=int, default=10000)
    return p


@requires_optional("dask.array", "xarray", "xarrayms",
                   da_import_error, xr_import_error, xms_import_error)
def predict(args):
    # Numpy arrays
    radec = np.array([[1.0, 1.2]]*10)
    stokes = np.array([[1.0, 0.0, 0.0, 0.0]]*10)

    # Dask arrays
    radec = da.from_array(radec, chunks=(3, 2))
    stokes = da.from_array(stokes, chunks=(3, 4))
    lm = radec_to_lm(radec)

    ddid_ds = list(xds_from_table('::'.join((args.ms, "DATA_DESCRIPTION")),
                                  group_cols="__row__"))

    spw_ds = list(xds_from_table('::'.join((args.ms, "SPECTRAL_WINDOW")),
                                 group_cols="__row__"))

    # List of write operations
    writes = []

    # Construct a graph for each DATA_DESC_ID
    for xds in xds_from_ms(args.ms,
                           columns=["UVW", "ANTENNA1", "ANTENNA2", "TIME"],
                           group_cols=["DATA_DESC_ID"],
                           chunks={"row": args.row_chunks}):

        # Extract frequencies from the spectral window associated
        # with this data descriptor id
        ddid = ddid_ds[xds.attrs['DATA_DESC_ID']]
        spw = spw_ds[ddid.SPECTRAL_WINDOW_ID.values]
        frequency = spw.CHAN_FREQ.data

        # (source, row, frequency)
        phase = phase_delay(lm, xds.UVW.data, frequency)
        # (source, corr1, corr2)
        brightness = stokes_convert(stokes, ["I", "Q", "U", "V"],
                                    [["XX", "XY"], ["YX", "YY"]])

        # (source, row, frequency, corr1, corr2)
        jones = da.einsum("srf, sij -> srfij", phase, brightness)

        # Identify time indices
        _, time_index = da.unique(xds.TIME.data, return_inverse=True)

        # Predict visibilities
        vis = predict_vis(time_index, xds.ANTENNA1.data, xds.ANTENNA2.data,
                          None, jones, None, None, None, None)

        # Reshape (2, 2) correlation to shape (4,)
        vis = vis.reshape(vis.shape[:2] + (4,))

        # Assign visibilities to MODEL_DATA array on the dataset
        model_data = xr.DataArray(vis, dims=["row", "chan", "corr"])
        xds = xds.assign(MODEL_DATA=model_data)
        # Create a write to the table
        write = xds_to_table(xds, args.ms, ['MODEL_DATA'])
        # Add to the list of writes
        writes.append(write)

    # Submit all graph computations in parallel
    dask.compute(writes)


if __name__ == "__main__":
    args = create_parser().parse_args()
    predict(args)
