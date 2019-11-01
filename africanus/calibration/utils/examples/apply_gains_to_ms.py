#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This example show how to apply gains to an ms in a chunked up way.
The gains should be stored in a .npy file and have the shape
expected by the corrupt_vis function.
It is assumed that the direction axis is ordered in the same way as
model_cols where model_cols is a comma separated string
"""

import numpy as np
from africanus.calibration.utils.dask import corrupt_vis
from africanus.calibration.utils import chunkify_rows
from daskms import xds_from_ms, xds_to_table
from pyrap.tables import table
import dask.array as da
from dask.diagnostics import ProgressBar

import argparse


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", help="Name of measurement set", type=str)
    p.add_argument("--model_cols", help="Comma separated string of "
                   "merasuturement set columns containing data "
                   "for each source", default='MODEL_DATA', type=str)
    p.add_argument("--data_col", help="Column where data lives. "
                   "Only used to get shape of data at this stage",
                   default='DATA', type=str)
    p.add_argument("--out_col", help="Where to write the corrupted data to. "
                   "Must exist in MS before writing to it.",
                   default='CORRECTED_DATA', type=str)
    p.add_argument("--gain_file", help=".npy file containing gains in format "
                   "(time, antenna, freq, source, corr). "
                   "See corrupt_vis docs.", type=str)
    p.add_argument("--utimes_per_chunk",  default=32, type=int,
                   help="Number of unique times in each chunk.")
    p.add_argument("--ncpu", help="The number of threads to use. "
                   "Default of zero means all", default=0, type=int)
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
row_chunks, tbin_idx, tbin_counts = chunkify_rows(time, args.utimes_per_chunk)
# convert to dask arrays
tbin_idx = da.from_array(tbin_idx, chunks=(args.utimes_per_chunk))
tbin_counts = da.from_array(tbin_counts, chunks=(args.utimes_per_chunk))

# get model column names
model_cols = args.model_cols.split(',')
n_dir = len(model_cols)

# append antenna columns
cols = []
cols.append('ANTENNA1')
cols.append('ANTENNA2')
cols.append(args.data_col)
for col in model_cols:
    cols.append(col)

# load in gains
jones = np.load(args.gain_file)
jones = jones.astype(np.complex64)
jones_shape = jones.shape
ndims = len(jones_shape)
jones = da.from_array(jones, chunks=(args.utimes_per_chunk,)
                      + jones_shape[1:])

# load data in in chunks and apply gains to each chunk
xds = xds_from_ms(args.ms, columns=cols, chunks={"row": row_chunks})[0]
vis = getattr(xds, args.data_col).data
ant1 = xds.ANTENNA1.data
ant2 = xds.ANTENNA2.data

model = []
for col in model_cols:
    model.append(getattr(xds, col).data)
model = da.stack(model, axis=2).rechunk({2: 3})

# reshape the correlation axis
if model.shape[-1] > 2:
    n_row, n_chan, n_dir, n_corr = model.shape
    model = model.reshape(n_row, n_chan, n_dir, 2, 2)
    reshape_vis = True
else:
    reshape_vis = False

# apply gains
corrupted_data = corrupt_vis(tbin_idx, tbin_counts, ant1, ant2,
                             jones, model)

if reshape_vis:
    corrupted_data = corrupted_data.reshape(n_row, n_chan, n_corr)

# Assign visibilities to args.out_col and write to ms
xds = xds.assign(**{args.out_col: (("row", "chan", "corr"), corrupted_data)})
# Create a write to the table
write = xds_to_table(xds, args.ms, [args.out_col])

# Submit all graph computations in parallel
with ProgressBar():
    write.compute()
