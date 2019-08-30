#!/usr/bin/env python
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
from africanus.calibration.utils.dask import compute_and_corrupt_vis
from africanus.calibration.utils import chunkify_rows
from africanus.dft.dask import im_to_vis
import xarray as xr
from xarrayms import xds_from_ms, xds_to_table
from pyrap.tables import table
import dask.array as da
from dask.diagnostics import ProgressBar

import argparse

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", help="Name of measurement set", type=str)
    p.add_argument("--model_file", help=".npy file containing the "
                   "time variable model in format [time, chan, source, corr].",
                   type=str)
    p.add_argument("--coord_file", type=str, help="file containing source "
                   "coordinates in format [time, source, l, m].")
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
                   "Default of zero means all", default=10, type=int)
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
n_time = tbin_idx.size

# get freqs
freqs = table(args.ms+'::SPECTRAL_WINDOW').getcol('CHAN_FREQ')[0]
n_freq = freqs.size
freqs = da.from_array(freqs, chunks=(n_freq))

# get source coordinates
lm = np.load(args.coord_file)
assert lm.shape[0] == n_time

# load in the model file
model = np.load(args.model_file)

assert model.shape[0] == n_time
assert model.shape[1] == n_freq

# get number of sources
n_dir = model.shape[2]
assert lm.shape[1] == n_dir

lm = da.from_array(lm, chunks=(args.utimes_per_chunk, n_dir, 2))

# append antenna columns
cols = []
cols.append('ANTENNA1')
cols.append('ANTENNA2')
cols.append('UVW')
cols.append(args.data_col)

# load in gains
jones = np.load(args.gain_file)
jones = jones.astype(np.complex64)
jones_shape = jones.shape
ndims = len(jones_shape)
jones = da.from_array(jones, chunks=(args.utimes_per_chunk,)
                      + jones_shape[1::])

# change model to dask array
model = da.from_array(model, chunks=(args.utimes_per_chunk,)
                      + model.shape[1::])

# load data in in chunks and apply gains to each chunk
xds = xds_from_ms(args.ms, columns=cols, chunks={"row": row_chunks})[0]
vis = getattr(xds, args.data_col).data
ant1 = xds.ANTENNA1.data
ant2 = xds.ANTENNA2.data
uvw = xds.UVW.data

# apply gains
print(jones.shape)
print(model.shape)
print(uvw.shape)
print(freqs.shape)
print(lm.shape)
corrupted_data = compute_and_corrupt_vis(tbin_idx, tbin_counts, ant1, ant2,
                                         jones, model, uvw, freqs, lm)

if reshape_vis:
    corrupted_data = corrupted_data.reshape(n_row, n_chan, n_corr)

# Assign visibilities to args.out_col and write to ms
data = xr.DataArray(corrupted_data, dims=["row", "chan", "corr"])

xds = xds.assign(**{args.out_col: data})

# Create a write to the table
write = xds_to_table(xds, args.ms, [args.out_col])

# Submit all graph computations in parallel
with ProgressBar():
    write.compute()
