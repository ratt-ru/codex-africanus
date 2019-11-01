# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from africanus.calibration.phase_only import gauss_newton
from africanus.calibration.utils import chunkify_rows
from africanus.calibration.phase_only import compute_jhj as np_compute_jhj
from africanus.calibration.phase_only import compute_jhr as np_compute_jhr


def test_compute_jhj_and_jhr(data_factory):
    # TODO - think of better tests for these
    # simulate noise free data with random DDE's
    n_dir = 1
    n_time = 32
    n_chan = 16
    n_ant = 7
    sigma_n = 0.0
    sigma_f = 0.05
    corr_shape = (2,)
    jones_shape = (2,)
    data_dict = data_factory(sigma_n, sigma_f, n_time, n_chan,
                             n_ant, n_dir, corr_shape, jones_shape)
    time = data_dict['TIME']
    _, time_bin_indices, time_bin_counts = chunkify_rows(time, n_time)
    ant1 = data_dict['ANTENNA1']
    ant2 = data_dict['ANTENNA2']
    vis = data_dict['DATA']
    model = data_dict['MODEL_DATA']
    jones = data_dict['JONES']
    flag = data_dict['FLAG']

    from africanus.calibration.phase_only.phase_only import compute_jhj
    from africanus.calibration.phase_only.phase_only import compute_jhr
    from africanus.calibration.phase_only.phase_only import compute_jhj_and_jhr
    jhj1, jhr1 = compute_jhj_and_jhr(time_bin_indices, time_bin_counts,
                                     ant1, ant2, jones, vis, model, flag)
    jhj2 = compute_jhj(time_bin_indices, time_bin_counts,
                       ant1, ant2, jones, model, flag)
    jhr2 = compute_jhr(time_bin_indices, time_bin_counts,
                       ant1, ant2, jones, vis, model, flag)

    assert_array_almost_equal(jhj1, jhj2, decimal=10)
    assert_array_almost_equal(jhr1, jhr2, decimal=10)


def test_compute_jhj_dask(data_factory):
    da = pytest.importorskip("dask.array")
    n_dir = 3
    n_time = 32
    n_chan = 16
    n_ant = 7
    sigma_n = 0.1
    sigma_f = 0.05
    corr_shape = (2,)
    jones_shape = (2,)
    data_dict = data_factory(sigma_n, sigma_f, n_time, n_chan,
                             n_ant, n_dir, corr_shape, jones_shape)
    time = data_dict['TIME']
    ncpu = 8
    utimes_per_chunk = n_time//ncpu
    row_chunks, time_bin_idx, time_bin_counts = chunkify_rows(
        time, utimes_per_chunk)
    ant1 = data_dict['ANTENNA1']
    ant2 = data_dict['ANTENNA2']
    model = data_dict['MODEL_DATA']
    jones = data_dict['JONES']
    flag = data_dict['FLAG']

    # get the numpy result
    jhj = np_compute_jhj(time_bin_idx, time_bin_counts, ant1, ant2,
                         jones, model, flag)

    da_time_bin_idx = da.from_array(time_bin_idx,
                                    chunks=(utimes_per_chunk))
    da_time_bin_counts = da.from_array(time_bin_counts,
                                       chunks=(utimes_per_chunk))
    da_ant1 = da.from_array(ant1, chunks=row_chunks)
    da_ant2 = da.from_array(ant2, chunks=row_chunks)
    da_model = da.from_array(model, chunks=(
        row_chunks, (n_chan,), (n_dir,)) + (corr_shape))
    da_jones = da.from_array(jones, chunks=(
        utimes_per_chunk, n_ant, n_chan, n_dir)+jones_shape)
    da_flag = da.from_array(flag, chunks=(row_chunks, (n_chan,)) + corr_shape)

    from africanus.calibration.phase_only.dask import compute_jhj

    da_jhj = compute_jhj(da_time_bin_idx, da_time_bin_counts,
                         da_ant1, da_ant2, da_jones, da_model, da_flag)

    jhj2 = da_jhj.compute()

    assert_array_almost_equal(jhj, jhj2, decimal=10)


def test_compute_jhr_dask(data_factory):
    da = pytest.importorskip("dask.array")
    n_dir = 3
    n_time = 32
    n_chan = 16
    n_ant = 7
    sigma_n = 0.1
    sigma_f = 0.05
    corr_shape = (2,)
    jones_shape = (2,)
    data_dict = data_factory(sigma_n, sigma_f, n_time, n_chan,
                             n_ant, n_dir, corr_shape, jones_shape)
    time = data_dict['TIME']
    ncpu = 8
    utimes_per_chunk = n_time//ncpu
    row_chunks, time_bin_idx, time_bin_counts = chunkify_rows(
        time, utimes_per_chunk)
    ant1 = data_dict['ANTENNA1']
    ant2 = data_dict['ANTENNA2']
    model = data_dict['MODEL_DATA']
    jones = data_dict['JONES']
    vis = data_dict['DATA']
    flag = data_dict['FLAG']

    # get the numpy result
    jhr = np_compute_jhr(time_bin_idx, time_bin_counts, ant1, ant2,
                         jones, vis, model, flag)

    da_time_bin_idx = da.from_array(time_bin_idx,
                                    chunks=(utimes_per_chunk))
    da_time_bin_counts = da.from_array(time_bin_counts,
                                       chunks=(utimes_per_chunk))
    da_ant1 = da.from_array(ant1, chunks=row_chunks)
    da_ant2 = da.from_array(ant2, chunks=row_chunks)
    da_model = da.from_array(model, chunks=(
        row_chunks, (n_chan,), (n_dir,)) + (corr_shape))
    da_jones = da.from_array(jones, chunks=(
        utimes_per_chunk, n_ant, n_chan, n_dir)+jones_shape)
    da_flag = da.from_array(flag, chunks=(row_chunks, (n_chan,)) + corr_shape)
    da_vis = da.from_array(vis, chunks=(row_chunks, (n_chan,)) + corr_shape)

    from africanus.calibration.phase_only.dask import compute_jhr

    da_jhr = compute_jhr(da_time_bin_idx, da_time_bin_counts,
                         da_ant1, da_ant2, da_jones, da_vis, da_model, da_flag)

    jhr2 = da_jhr.compute()

    assert_array_almost_equal(jhr, jhr2, decimal=10)


def test_phase_only_diag_diag(data_factory):
    """
    Test phase only calibration by checking that
    we reconstruct the correct gains for a noise
    free simulation.
    """
    np.random.seed(420)
    # simulate noise free data with random DDE's
    n_dir = 3
    n_time = 32
    n_chan = 16
    n_ant = 7
    sigma_n = 0.0
    sigma_f = 0.1
    corr_shape = (2,)
    jones_shape = (2,)
    data_dict = data_factory(sigma_n, sigma_f, n_time, n_chan,
                             n_ant, n_dir, corr_shape, jones_shape,
                             phase_only_gains=True)
    time = data_dict['TIME']
    _, time_bin_indices, time_bin_counts = chunkify_rows(time, n_time)
    ant1 = data_dict['ANTENNA1']
    ant2 = data_dict['ANTENNA2']
    vis = data_dict['DATA']
    model = data_dict['MODEL_DATA']
    jones = data_dict['JONES']
    flag = data_dict['FLAG']
    weight = data_dict['WEIGHT_SPECTRUM']
    # calibrate the data
    jones0 = np.ones((n_time, n_ant, n_chan, n_dir) + jones_shape,
                     dtype=np.complex128)
    precision = 5
    gains, jhj, jhr, k = gauss_newton(
        time_bin_indices, time_bin_counts,
        ant1, ant2, jones0, vis,
        flag, model, weight,
        tol=10**(-precision), maxiter=250)
    # check that phase differences are correct
    for p in range(n_ant):
        for q in range(p):
            phase_diff_true = np.angle(jones[:, p]) - np.angle(jones[:, q])
            phase_diff = np.angle(gains[:, p]) - np.angle(gains[:, q])
            assert_array_almost_equal(
                phase_diff_true, phase_diff, decimal=precision-3)
