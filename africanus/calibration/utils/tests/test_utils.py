# -*- coding: utf-8 -*-

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from africanus.averaging.support import unique_time
from africanus.calibration.utils import chunkify_rows
from africanus.rime.predict import predict_vis

corr_shape_parametrization = pytest.mark.parametrize(
    'corr_shape, jones_shape',
    [((1,), (1,)),  # DIAG_DIAG
     ((2,), (2,)),  # DIAG_DIAG
     ((2, 2), (2,)),  # DIAG
     ((2, 2), (2, 2)),  # FULL
     ])


@corr_shape_parametrization
def test_corrupt_vis(data_factory, corr_shape, jones_shape):
    """
    Tests corrupt vis against predict_vis. They should do
    the same thing but corrupt_vis adheres to the structure
    in the africanus.calibration module.
    """
    # simulate noise free data with random DDE's
    n_dir = 3
    n_time = 32
    n_chan = 16
    n_ant = 7
    sigma_n = 0.0
    sigma_f = 0.05
    data_dict = data_factory(sigma_n, sigma_f, n_time, n_chan,
                             n_ant, n_dir, corr_shape, jones_shape)
    # make_data uses corrupt_vis to produce the data so we only need to test
    # that predict vis gives the same thing on the reshaped arrays
    ant1 = data_dict['ANTENNA1']
    ant2 = data_dict['ANTENNA2']
    vis = data_dict['DATA']
    model = data_dict['MODEL_DATA']
    jones = data_dict['JONES']
    time = data_dict['TIME']

    # predict_vis expects (source, time, ant, chan, corr1, corr2) so
    # we need to transpose the axes while preserving corr_shape and jones_shape
    if jones_shape != corr_shape:
        # This only happens in DIAG mode and we need to broadcast jones_shape
        # to match corr_shape
        tmp = np.zeros((n_time, n_ant, n_chan, n_dir) +
                       corr_shape, dtype=np.complex128)
        tmp[:, :, :, :, 0, 0] = jones[:, :, :, :, 0]
        tmp[:, :, :, :, 1, 1] = jones[:, :, :, :, 1]
        jones = tmp

    if len(corr_shape) == 2:
        jones = np.transpose(jones, [3, 0, 1, 2, 4, 5])
        model = np.transpose(model, [2, 0, 1, 3, 4])
    elif len(corr_shape) == 1:
        jones = np.transpose(jones, [3, 0, 1, 2, 4])
        model = np.transpose(model, [2, 0, 1, 3])
    else:
        raise ValueError("Unsupported correlation shapes")

    # get vis
    time_index = np.unique(time, return_inverse=True)[1]
    test_vis = predict_vis(time_index, ant1, ant2,
                           source_coh=model,
                           dde1_jones=jones,
                           dde2_jones=jones)

    assert_array_almost_equal(test_vis, vis, decimal=10)


@corr_shape_parametrization
def test_residual_vis(data_factory, corr_shape, jones_shape):
    """
    Tests subtraction of model by subtracting all but one
    direction from noise free simulated data and comparing
    the output to the unsubtracted direction.
    """
    from africanus.calibration.utils import residual_vis, corrupt_vis
    # simulate noise free data with random DDE's
    n_dir = 3
    n_time = 32
    n_chan = 16
    n_ant = 7
    sigma_n = 0.0
    sigma_f = 0.05
    data_dict = data_factory(sigma_n, sigma_f, n_time, n_chan,
                             n_ant, n_dir, corr_shape, jones_shape)
    time = data_dict['TIME']
    _, time_bin_indices, _, time_bin_counts = unique_time(time)
    ant1 = data_dict['ANTENNA1']
    ant2 = data_dict['ANTENNA2']
    vis = data_dict['DATA']
    model = data_dict['MODEL_DATA']
    jones = data_dict['JONES']
    flag = data_dict['FLAG']
    # split the model and jones terms
    model_unsubtracted = model[:, :, 0:1]
    model_subtract = model[:, :, 1::]
    jones_unsubtracted = jones[:, :, :, 0:1]
    jones_subtract = jones[:, :, :, 1::]
    # subtract all but one direction
    residual = residual_vis(time_bin_indices, time_bin_counts,
                            ant1, ant2, jones_subtract, vis,
                            flag, model_subtract)
    # apply gains to the unsubtracted direction
    vis_unsubtracted = corrupt_vis(time_bin_indices,
                                   time_bin_counts,
                                   ant1, ant2,
                                   jones_unsubtracted,
                                   model_unsubtracted)
    # residual should now be equal to unsubtracted vis
    assert_array_almost_equal(residual, vis_unsubtracted, decimal=10)


@corr_shape_parametrization
def test_correct_vis(data_factory, corr_shape, jones_shape):
    """
    Tests correct_vis by correcting noise free simulation
    with random DIE gains
    """
    from africanus.calibration.utils import correct_vis
    # simulate noise free data with only DIE's
    n_dir = 1
    n_time = 32
    n_chan = 16
    n_ant = 7
    sigma_n = 0.0
    sigma_f = 0.05
    data_dict = data_factory(sigma_n, sigma_f, n_time,
                             n_chan, n_ant, n_dir, corr_shape,
                             jones_shape)
    time = data_dict['TIME']
    _, time_bin_indices, _, time_bin_counts = unique_time(time)
    ant1 = data_dict['ANTENNA1']
    ant2 = data_dict['ANTENNA2']
    vis = data_dict['DATA']
    model = data_dict['MODEL_DATA']
    jones = data_dict['JONES']
    flag = data_dict['FLAG']
    # correct vis
    corrected_vis = correct_vis(
        time_bin_indices, time_bin_counts,
        ant1, ant2, jones, vis, flag)
    # squeeze out dir axis to get expected model data
    model = model.reshape(vis.shape)
    assert_array_almost_equal(corrected_vis, model, decimal=10)


@corr_shape_parametrization
def test_corrupt_vis_dask(data_factory, corr_shape, jones_shape):
    da = pytest.importorskip("dask.array")
    # simulate noise free data with only DIE's
    n_dir = 3
    n_time = 32
    n_chan = 16
    n_ant = 4
    sigma_n = 0.0
    sigma_f = 0.05
    data_dict = data_factory(sigma_n, sigma_f, n_time,
                             n_chan, n_ant, n_dir, corr_shape,
                             jones_shape)
    vis = data_dict['DATA']  # what we need to compare to
    ant1 = data_dict['ANTENNA1']
    ant2 = data_dict['ANTENNA2']
    model = data_dict['MODEL_DATA']
    jones = data_dict['JONES']
    time = data_dict['TIME']

    # get chunking scheme
    ncpu = 8
    utimes_per_chunk = n_time//ncpu
    row_chunks, time_bin_idx, time_bin_counts = chunkify_rows(
        time, utimes_per_chunk)

    # set up dask arrays
    da_time_bin_idx = da.from_array(time_bin_idx, chunks=(utimes_per_chunk))
    da_time_bin_counts = da.from_array(
        time_bin_counts, chunks=(utimes_per_chunk))
    da_ant1 = da.from_array(ant1, chunks=row_chunks)
    da_ant2 = da.from_array(ant2, chunks=row_chunks)
    da_model = da.from_array(model, chunks=(
        row_chunks, (n_chan,), (n_dir,)) + (corr_shape))
    da_jones = da.from_array(jones, chunks=(
        utimes_per_chunk, n_ant, n_chan, n_dir)+jones_shape)

    from africanus.calibration.utils.dask import corrupt_vis
    da_vis = corrupt_vis(da_time_bin_idx, da_time_bin_counts,
                         da_ant1, da_ant2, da_jones, da_model)
    vis2 = da_vis.compute()
    assert_array_almost_equal(vis, vis2, decimal=10)


@corr_shape_parametrization
def test_correct_vis_dask(data_factory, corr_shape, jones_shape):
    da = pytest.importorskip("dask.array")
    # simulate noise free data with only DIE's
    n_dir = 1
    n_time = 32
    n_chan = 16
    n_ant = 4
    sigma_n = 0.0
    sigma_f = 0.05
    data_dict = data_factory(sigma_n, sigma_f, n_time,
                             n_chan, n_ant, n_dir, corr_shape,
                             jones_shape)
    vis = data_dict['DATA']
    ant1 = data_dict['ANTENNA1']
    ant2 = data_dict['ANTENNA2']
    model = data_dict['MODEL_DATA']  # what we need to compare to
    jones = data_dict['JONES']
    time = data_dict['TIME']
    flag = data_dict['FLAG']

    # get chunking scheme
    ncpu = 8
    utimes_per_chunk = n_time//ncpu
    row_chunks, time_bin_idx, time_bin_counts = chunkify_rows(
        time, utimes_per_chunk)

    # set up dask arrays
    da_time_bin_idx = da.from_array(time_bin_idx, chunks=(utimes_per_chunk))
    da_time_bin_counts = da.from_array(
        time_bin_counts, chunks=(utimes_per_chunk))
    da_ant1 = da.from_array(ant1, chunks=row_chunks)
    da_ant2 = da.from_array(ant2, chunks=row_chunks)
    da_vis = da.from_array(vis, chunks=(row_chunks, (n_chan,)) + (corr_shape))
    da_jones = da.from_array(jones, chunks=(
        utimes_per_chunk, n_ant, n_chan, n_dir)+jones_shape)
    da_flag = da.from_array(flag, chunks=(
        row_chunks, (n_chan,)) + (corr_shape))

    from africanus.calibration.utils.dask import correct_vis
    da_model = correct_vis(da_time_bin_idx, da_time_bin_counts, da_ant1,
                           da_ant2, da_jones, da_vis, da_flag)
    model2 = da_model.compute()
    assert_array_almost_equal(model.reshape(model2.shape), model2, decimal=10)


@corr_shape_parametrization
def test_residual_vis_dask(data_factory, corr_shape, jones_shape):
    da = pytest.importorskip("dask.array")
    # simulate noise free data with only DIE's
    n_dir = 3
    n_time = 32
    n_chan = 16
    n_ant = 4
    sigma_n = 0.0
    sigma_f = 0.05
    data_dict = data_factory(sigma_n, sigma_f, n_time,
                             n_chan, n_ant, n_dir, corr_shape,
                             jones_shape)
    vis = data_dict['DATA']
    ant1 = data_dict['ANTENNA1']
    ant2 = data_dict['ANTENNA2']
    model = data_dict['MODEL_DATA']  # what we need to compare to
    jones = data_dict['JONES']
    time = data_dict['TIME']
    flag = data_dict['FLAG']

    # get chunking scheme
    ncpu = 8
    utimes_per_chunk = n_time//ncpu
    row_chunks, time_bin_idx, time_bin_counts = chunkify_rows(
        time, utimes_per_chunk)

    # set up dask arrays
    da_time_bin_idx = da.from_array(time_bin_idx, chunks=(utimes_per_chunk))
    da_time_bin_counts = da.from_array(
        time_bin_counts, chunks=(utimes_per_chunk))
    da_ant1 = da.from_array(ant1, chunks=row_chunks)
    da_ant2 = da.from_array(ant2, chunks=row_chunks)
    da_vis = da.from_array(vis, chunks=(row_chunks, (n_chan,)) + (corr_shape))
    da_model = da.from_array(model, chunks=(
        row_chunks, (n_chan,), (n_dir,)) + (corr_shape))
    da_jones = da.from_array(jones, chunks=(
        utimes_per_chunk, n_ant, n_chan, n_dir)+jones_shape)
    da_flag = da.from_array(flag, chunks=(
        row_chunks, (n_chan,)) + (corr_shape))

    from africanus.calibration.utils import residual_vis as residual_vis_np
    residual = residual_vis_np(time_bin_idx, time_bin_counts, ant1, ant2,
                               jones, vis, flag, model)

    from africanus.calibration.utils.dask import residual_vis
    da_residual = residual_vis(da_time_bin_idx, da_time_bin_counts,
                               da_ant1, da_ant2, da_jones, da_vis,
                               da_flag, da_model)
    residual2 = da_residual.compute()
    assert_array_almost_equal(residual, residual2, decimal=10)
