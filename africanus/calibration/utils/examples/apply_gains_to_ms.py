# -*- coding: utf-8 -*-
"""
This example show how to apply gains to an ms in a chunked up way.
The gains should be stored in a .npy file and have the shape
expected by the corrupt_vis function. 
It is assumed that the direction axis is ordered in the same way as
model_cols where model_cols is a comma separated string
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from africanus.calibration.utils.dask import corrupt_vis
from africanus.calibration.utils import chunkify_rows
import xarray as xr
from xarrayms import xds_from_ms, xds_from_table, xds_to_table
from pyrap.tables import table
import dask.array as da
from dask.diagnostics import ProgressBar
from africanus.calibration.utils import check_type

import argparse

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", type=str)
    p.add_argument("--model_cols", default='MODEL_DATA', type=str)
    p.add_argument("--out_col", default='CORRUPTED_DATA', type=str)
    p.add_argument("--gain_file", type=str)
    p.add_argument("--utime_per_chunk", default=32, type=int)
    p.add_argument("--ncpu", default=0, type=int)
    p.add_argument('--field', default=0, type=int)
    return p

args = create_parser().parse_args()

if args.ncpu:
    ncpu = args.ncpu
    from multiprocessing.pool import ThreadPool
    import dask
    dask.config.set(pool=ThreadPool(ncpu))
else:
    import multiprocessing
    ncpu = multiprocessing.cpu_count()

print("Using %i threads" % ncpu)

# get full time column and compute row chunks
time = table(args.ms).getcol('TIME')
row_chunks, time_bin_idx, time_bin_counts = chunkify_rows(time, args.utimes_per_chunk)
# convert to dask arrays
time_bin_idx = da.from_array(time_bin_idx, chunks=(args.utimes_per_chunk))
time_bin_counts = da.from_array(time_bin_counts, chunks=(args.utimes_per_chunk))

# get model column names
model_cols = args.model_cols.split(',')
n_dir = len(model_cols)

# append antenna columns
cols = model_cols
cols.append('ANTENNA1')
cols.append('ANTENNA2')

# load in gains
jones = np.load(args.gains_file)
jones_shape = jones.shape
ndims = len(jones_shape)
jones = da.from_array(jones, chunks=(args.utime_per_chunk,) + jones_shape[1:])

# load data in in chunks and apply gains to each chunk
writes = []
for xds in xds_from_ms(args.ms,
                       columns=cols,
                       chunks={"row": row_chunks}):
    vis = getattr(xds, args.data_col).data
    ant1 = xds.ANTENNA1.data
    ant2 = xds.ANTENNA2.data
    model_shape = vis.shape[0:2] + (n_dir,) + vis.shape[2:]
    model = da.empty(model_shape, vis.dtype)
    for d, col in enumerate(model_cols):
        model[:, :, d] = getattr(xds, col).data
    
    # apply gains
    corrupted_data = corrupt_vis(time_bin_idx, time_bin_counts, ant1, ant2,
                                 jones, model)
    
    # Assign visibilities to args.out_col and write to ms
    mode = check_type(jones, vis)
    if mode == 0:
        data = xr.DataArray(corrupted_data, dims=["row", "chan", "corr"])
    else:
        data = xr.DataArray(corrupted_data, dims=["row", "chan", "corr", "corr"])
    xds = xds.assign(**{args.out_col: data})
    # Create a write to the table
    write = xds_to_table(xds, args.ms, [args.out_col])
    # Add to the list of writes
    writes.append(write)

# Submit all graph computations in parallel
with ProgressBar():
    dask.compute(writes)
