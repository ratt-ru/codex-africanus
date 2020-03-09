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
from africanus.calibration.utils.dask import compute_and_corrupt_vis
from africanus.calibration.utils import chunkify_rows
from daskms import xds_from_ms, xds_to_table
from pyrap.tables import table
import dask.array as da
from dask.diagnostics import ProgressBar
import Tigger
from africanus.coordinates import radec_to_lm
import argparse


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", help="Name of measurement set", type=str)
    p.add_argument("--sky_model", type=str, help="Tigger lsm file")
    p.add_argument("--data_col", help="Column where data lives. "
                   "Only used to get shape of data at this stage",
                   default='DATA', type=str)
    p.add_argument("--out_col", help="Where to write the corrupted data to. "
                   "Must exist in MS before writing to it.",
                   default='DATA', type=str)
    p.add_argument("--gain_file", help=".npy file containing gains in format "
                   "(time, antenna, freq, source, corr). "
                   "See corrupt_vis docs.", type=str)
    p.add_argument("--utimes_per_chunk",  default=32, type=int,
                   help="Number of unique times in each chunk.")
    p.add_argument("--ncpu", help="The number of threads to use. "
                   "Default of zero means all", default=10, type=int)
    p.add_argument('--field', default=0, type=int)
    return p


def main(args):
    # get full time column and compute row chunks
    ms = table(args.ms)
    time = ms.getcol('TIME')
    row_chunks, tbin_idx, tbin_counts = chunkify_rows(
        time, args.utimes_per_chunk)
    # convert to dask arrays
    tbin_idx = da.from_array(tbin_idx, chunks=(args.utimes_per_chunk))
    tbin_counts = da.from_array(tbin_counts, chunks=(args.utimes_per_chunk))
    n_time = tbin_idx.size
    ms.close()

    # get phase dir
    fld = table(args.ms+'::FIELD')
    radec0 = fld.getcol('PHASE_DIR').squeeze().reshape(1, 2)
    radec0 = np.tile(radec0, (n_time, 1))
    fld.close()

    # get freqs
    freqs = table(
        args.ms+'::SPECTRAL_WINDOW').getcol('CHAN_FREQ')[0].astype(np.float64)
    n_freq = freqs.size
    freqs = da.from_array(freqs, chunks=(n_freq))

    # get source coordinates from lsm
    lsm = Tigger.load(args.sky_model)
    radec = []
    stokes = []
    spi = []
    ref_freqs = []

    for source in lsm.sources:
        radec.append([source.pos.ra, source.pos.dec])
        stokes.append([source.flux.I])
        spi.append(source.spectrum.spi)
        ref_freqs.append(source.spectrum.freq0)

    n_dir = len(stokes)
    radec = np.asarray(radec)
    lm = np.zeros((n_time,) + radec.shape)
    for t in range(n_time):
        lm[t] = radec_to_lm(radec, radec0[t])

    lm = da.from_array(lm, chunks=(args.utimes_per_chunk, n_dir, 2))

    # load in the model file
    n_corr = 1
    model = np.zeros((n_time, n_freq, n_dir, n_corr))
    stokes = np.asarray(stokes)
    ref_freqs = np.asarray(ref_freqs)
    spi = np.asarray(spi)
    for t in range(n_time):
        for d in range(n_dir):
            model[t, :, d, 0] = stokes[d] * (freqs/ref_freqs[d])**spi[d]

    # append antenna columns
    cols = []
    cols.append('ANTENNA1')
    cols.append('ANTENNA2')
    cols.append('UVW')

    # load in gains
    jones = np.load(args.gain_file)
    jones = jones.astype(np.complex128)
    jones_shape = jones.shape
    jones = da.from_array(jones, chunks=(args.utimes_per_chunk,)
                          + jones_shape[1::])

    # change model to dask array
    model = da.from_array(model, chunks=(args.utimes_per_chunk,)
                          + model.shape[1::])

    # load data in in chunks and apply gains to each chunk
    xds = xds_from_ms(args.ms, columns=cols, chunks={"row": row_chunks})[0]
    ant1 = xds.ANTENNA1.data
    ant2 = xds.ANTENNA2.data
    uvw = xds.UVW.data

    # apply gains
    data = compute_and_corrupt_vis(tbin_idx, tbin_counts, ant1, ant2,
                                   jones, model, uvw, freqs, lm)

    # Assign visibilities to args.out_col and write to ms
    xds = xds.assign(**{args.out_col: (("row", "chan", "corr"), data)})
    # Create a write to the table
    write = xds_to_table(xds, args.ms, [args.out_col])

    # Submit all graph computations in parallel
    with ProgressBar():
        write.compute()


if __name__ == "__main__":
    args = create_parser().parse_args()

    if args.ncpu:
        from multiprocessing.pool import ThreadPool
        import dask
        dask.config.set(pool=ThreadPool(args.ncpu))
    else:
        import multiprocessing
        args.ncpu = multiprocessing.cpu_count()

    print("Using %i threads" % args.ncpu)

    main(args)
