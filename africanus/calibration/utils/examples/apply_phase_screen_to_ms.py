# -*- coding: utf-8 -*-
"""
This example show how to apply gains to an ms in a chunked up way.
The gains should be stored in a .npy file and have the shape
expected by the corrupt_vis function.
It is assumed that the direction axis is ordered in the same way as
model_cols where model_cols is a comma separated string
"""
from time import time as timeit
from numpy.testing import assert_array_almost_equal
import argparse
from africanus.coordinates import radec_to_lm
import Tigger
from dask.diagnostics import ProgressBar
import dask.array as da
from pyrap.tables import table
from daskms import xds_from_ms, xds_to_table
from africanus.dft import im_to_vis
from africanus.calibration.phase_only import gauss_newton
from africanus.calibration.utils import chunkify_rows
from africanus.calibration.utils.dask import compute_and_corrupt_vis
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", help="Name of measurement set", type=str)
    p.add_argument("--sky_model", type=str, help="Tigger lsm file")
    p.add_argument("--out_col", help="Where to write the corrupted data to. "
                   "Must exist in MS before writing to it.",
                   default='DATA', type=str)
    p.add_argument("--utimes_per_chunk",  default=64, type=int,
                   help="Number of unique times in each chunk.")
    p.add_argument("--ncpu", help="The number of threads to use. "
                   "Default of zero means all", default=10, type=int)
    return p


def make_screen(lm, freq, n_time, n_ant, n_corr):
    n_dir = lm.shape[0]
    n_freq = freq.size
    # create basis matrix for plane [1, l, m]
    n_coeff = 3
    l_coord = lm[:, 0]
    m_coord = lm[:, 1]
    basis = np.hstack((np.ones((n_dir, 1), dtype=np.float64),
                       l_coord[:, None], m_coord[:, None]))
    # get coeffs
    alphas = 0.05 * np.random.randn(n_time, n_ant, n_coeff, n_corr)
    # normalise freqs
    freq_norm = freq/freq.max()
    # simulate phases
    phases = np.zeros((n_time, n_ant, n_freq, n_dir, n_corr),
                      dtype=np.float64)
    for t in range(n_time):
        for p in range(n_ant):
            for c in range(n_corr):
                # get screen at source locations
                screen = basis.dot(alphas[t, p, :, c])
                # apply frequency scaling
                phases[t, p, :, :, c] = screen[None, :]/freq_norm[:, None]
    return np.exp(1.0j*phases), alphas


def simulate(args):
    # get full time column and compute row chunks
    ms = table(args.ms)
    time = ms.getcol('TIME')
    row_chunks, tbin_idx, tbin_counts = chunkify_rows(
        time, args.utimes_per_chunk)
    # convert to dask arrays
    tbin_idx = da.from_array(tbin_idx, chunks=(args.utimes_per_chunk))
    tbin_counts = da.from_array(tbin_counts, chunks=(args.utimes_per_chunk))
    n_time = tbin_idx.size
    ant1 = ms.getcol('ANTENNA1')
    ant2 = ms.getcol('ANTENNA2')
    n_ant = np.maximum(ant1.max(), ant2.max()) + 1
    flag = ms.getcol("FLAG")
    n_row, n_freq, n_corr = flag.shape
    if n_corr == 4:
        model_corr = (2, 2)
        jones_corr = (2,)
    elif n_corr == 2:
        model_corr = (2,)
        jones_corr = (2,)
    elif n_corr == 1:
        model_corr = (1,)
        jones_corr = (1,)
    else:
        raise RuntimeError("Invalid number of correlations")
    ms.close()

    # get phase dir
    radec0 = table(args.ms+'::FIELD').getcol('PHASE_DIR').squeeze()

    # get freqs
    freq = table(
        args.ms+'::SPECTRAL_WINDOW').getcol('CHAN_FREQ')[0].astype(np.float64)
    assert freq.size == n_freq

    # get source coordinates from lsm
    lsm = Tigger.load(args.sky_model)
    radec = []
    stokes = []
    spi = []
    ref_freqs = []

    for source in lsm.sources:
        radec.append([source.pos.ra, source.pos.dec])
        stokes.append([source.flux.I])
        tmp_spec = source.spectrum
        spi.append([tmp_spec.spi if tmp_spec is not None else 0.0])
        ref_freqs.append([tmp_spec.freq0 if tmp_spec is not None else 1.0])

    n_dir = len(stokes)
    radec = np.asarray(radec)
    lm = radec_to_lm(radec, radec0)

    # load in the model file
    model = np.zeros((n_freq, n_dir) + model_corr)
    stokes = np.asarray(stokes)
    ref_freqs = np.asarray(ref_freqs)
    spi = np.asarray(spi)
    for d in range(n_dir):
        Stokes_I = stokes[d] * (freq/ref_freqs[d])**spi[d]
        if n_corr == 4:
            model[:, d, 0, 0] = Stokes_I
            model[:, d, 1, 1] = Stokes_I
        elif n_corr == 2:
            model[:, d, 0] = Stokes_I
            model[:, d, 1] = Stokes_I
        else:
            model[:, d, 0] = Stokes_I

    # append antenna columns
    cols = []
    cols.append('ANTENNA1')
    cols.append('ANTENNA2')
    cols.append('UVW')

    # load in gains
    jones, alphas = make_screen(lm, freq, n_time, n_ant, jones_corr[0])
    jones = jones.astype(np.complex128)
    jones_shape = jones.shape
    jones_da = da.from_array(jones, chunks=(args.utimes_per_chunk,)
                             + jones_shape[1::])

    freqs = da.from_array(freq, chunks=(n_freq))
    lm = da.from_array(np.tile(lm[None], (n_time, 1, 1)), chunks=(
        args.utimes_per_chunk, n_dir, 2))
    # change model to dask array
    tmp_shape = (n_time,)
    for i in range(len(model.shape)):
        tmp_shape += (1,)
    model = da.from_array(np.tile(model[None], tmp_shape),
                          chunks=(args.utimes_per_chunk,) + model.shape)

    # load data in in chunks and apply gains to each chunk
    xds = xds_from_ms(args.ms, columns=cols, chunks={"row": row_chunks})[0]
    ant1 = xds.ANTENNA1.data
    ant2 = xds.ANTENNA2.data
    uvw = xds.UVW.data

    # apply gains
    data = compute_and_corrupt_vis(tbin_idx, tbin_counts, ant1, ant2,
                                   jones_da, model, uvw, freqs, lm)

    # Assign visibilities to args.out_col and write to ms
    xds = xds.assign(**{args.out_col: (("row", "chan", "corr"),
                                       data.reshape(n_row, n_freq, n_corr))})
    # Create a write to the table
    write = xds_to_table(xds, args.ms, [args.out_col])

    # Submit all graph computations in parallel
    with ProgressBar():
        write.compute()

    return jones, alphas


def calibrate(args, jones, alphas):
    # simple calibration to test if simulation went as expected.
    # Note do not run on large data set

    # load data
    ms = table(args.ms)
    time = ms.getcol('TIME')
    _, tbin_idx, tbin_counts = chunkify_rows(time, args.utimes_per_chunk)
    n_time = tbin_idx.size
    ant1 = ms.getcol('ANTENNA1')
    ant2 = ms.getcol('ANTENNA2')
    n_ant = np.maximum(ant1.max(), ant2.max()) + 1
    uvw = ms.getcol('UVW').astype(np.float64)
    data = ms.getcol(args.out_col)  # this is where we put the data
    # we know it is pure Stokes I so we can solve using diagonals only
    data = data[:, :, (0, 3)].astype(np.complex128)
    n_row, n_freq, n_corr = data.shape
    flag = ms.getcol('FLAG')
    flag = flag[:, :, (0, 3)]

    # get phase dir
    radec0 = table(
        args.ms+'::FIELD').getcol('PHASE_DIR').squeeze().astype(np.float64)

    # get freqs
    freq = table(
        args.ms+'::SPECTRAL_WINDOW').getcol('CHAN_FREQ')[0].astype(np.float64)
    assert freq.size == n_freq

    # now get the model
    # get source coordinates from lsm
    lsm = Tigger.load(args.sky_model)
    radec = []
    stokes = []
    spi = []
    ref_freqs = []

    for source in lsm.sources:
        radec.append([source.pos.ra, source.pos.dec])
        stokes.append([source.flux.I])
        tmp_spec = source.spectrum
        spi.append([tmp_spec.spi if tmp_spec is not None else 0.0])
        ref_freqs.append([tmp_spec.freq0 if tmp_spec is not None else 1.0])

    n_dir = len(stokes)
    radec = np.asarray(radec)
    lm = radec_to_lm(radec, radec0)

    # get model visibilities
    model = np.zeros((n_row, n_freq, n_dir, 2), dtype=np.complex)
    stokes = np.asarray(stokes)
    ref_freqs = np.asarray(ref_freqs)
    spi = np.asarray(spi)
    for d in range(n_dir):
        Stokes_I = stokes[d] * (freq/ref_freqs[d])**spi[d]
        model[:, :, d, 0:1] = im_to_vis(
            Stokes_I[None, :, None], uvw, lm[d:d+1], freq)
        model[:, :, d, 1] = model[:, :, d, 0]

    # set weights to unity
    weight = np.ones_like(data, dtype=np.float64)

    # initialise gains
    jones0 = np.ones((n_time, n_ant, n_freq, n_dir, n_corr),
                     dtype=np.complex128)

    # calibrate
    ti = timeit()
    jones_hat, jhj, jhr, k = gauss_newton(
        tbin_idx, tbin_counts, ant1, ant2, jones0, data, flag, model,
        weight, tol=1e-5, maxiter=100)
    print("%i iterations took %fs" % (k, timeit() - ti))

    # verify result
    for p in range(2):
        for q in range(p):
            diff_true = np.angle(jones[:, p] * jones[:, q].conj())
            diff_hat = np.angle(jones_hat[:, p] * jones_hat[:, q].conj())
            try:
                assert_array_almost_equal(diff_true, diff_hat, decimal=2)
            except Exception as e:
                print(e)


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

    jones, alphas = simulate(args)

    calibrate(args, jones, alphas)
