# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import dask.array as da
from numpy.random import normal
import pytest
from numpy.testing import assert_array_almost_equal
from africanus.dft import im_to_vis
from africanus.averaging.support import unique_time
from africanus.calibration.utils import residual_vis, correct_vis, corrupt_vis
from africanus.calibration.utils import chunkify_rows
from africanus.rime.predict import predict_vis
from numba.types.misc import literal


def give_lm(n_dir):
    ls = 0.1*np.random.randn(n_dir)
    ms = 0.1*np.random.randn(n_dir)
    lm = np.vstack((ls, ms)).T
    return lm


def give_flux(n_dir, n_chan, cor_shape, alpha, freq, freq0):
    w = freq/freq0
    flux = np.zeros((n_dir, n_chan) + cor_shape, dtype=np.float64)
    for d in range(n_dir):
        tmp_flux = np.random.random(size=cor_shape)
        for v in range(n_chan):
            flux[d, v] = tmp_flux * w[v]**alpha
    return flux


corr_shape_parametrization = pytest.mark.parametrize(
    'corr_shape, jones_shape, mode',
    [((2,), (2,), literal(0)),  # DIAG_DIAG
     ((2, 2), (2,), literal(1)),  # DIAG
     ((2, 2), (2, 2), literal(2)),  # FULL
     ])


def make_data(sigma_n, sigma_f, n_time, n_chan, n_ant,
              n_dir, corr_shape, jones_shape, mode):
    n_bl = n_ant*(n_ant-1)//2
    n_row = n_bl*n_time
    # make aux data
    antenna1 = np.zeros(n_row, dtype=np.int16)
    antenna2 = np.zeros(n_row, dtype=np.int16)
    time = np.zeros(n_row, dtype=np.float64)
    uvw = np.zeros((n_row, 3), dtype=np.float64)
    time_values = np.linspace(0, 1, n_time)
    freq = np.linspace(1e9, 2e9, n_chan)
    for i in range(n_time):
        row = 0
        for p in range(n_ant):
            for q in range(p):
                time[i*n_bl + row] = time_values[i]
                antenna1[i*n_bl + row] = p
                antenna2[i*n_bl + row] = q
                uvw[i*n_bl + row] = np.random.randn(3)
                row += 1
    assert time.size == n_row
    # simulate visibilities
    model_data = np.zeros((n_row, n_chan, n_dir) +
                          corr_shape, dtype=np.complex128)
    # make up some sources
    lm = give_lm(n_dir)
    alpha = -0.7
    freq0 = freq[n_chan//2]
    flux = give_flux(n_dir, n_chan, corr_shape, alpha, freq, freq0)
    # simulate model data
    for dir in range(n_dir):
        dir_lm = lm[dir].reshape(1, 2)
        # Get flux for source (keep source axis, flatten cor axis)
        dir_flux = flux[dir].reshape(1, n_chan, np.prod(corr_shape))
        tmp = im_to_vis(dir_flux, uvw, dir_lm, freq)
        model_data[:, :, dir] = tmp.reshape((n_row, n_chan) + corr_shape)
    assert not np.isnan(model_data).any()
    # simulate gains (just randomly scattered around 1 for now)
    jones = np.ones((n_time, n_ant, n_chan, n_dir) +
                    jones_shape, dtype=np.complex128)
    if sigma_f:
        jones += (normal(loc=0.0, scale=sigma_f, size=jones.shape) +
                  1.0j*normal(loc=0.0, scale=sigma_f, size=jones.shape))
        assert (np.abs(jones) > 1e-5).all()
        assert not np.isnan(jones).any()
    # get vis
    _, time_bin_indices, _, time_bin_counts = unique_time(time)
    vis = np.zeros((n_row, n_chan) + corr_shape, dtype=np.complex128)
    vis = corrupt_vis(time_bin_indices, time_bin_counts,
                      antenna1, antenna2, jones, vis, model_data, mode)
    assert not np.isnan(vis).any()
    # add noise
    if sigma_n:
        vis += (normal(loc=0.0, scale=sigma_n, size=vis.shape) +
                1.0j*normal(loc=0.0, scale=sigma_n, size=vis.shape))
    weights = np.ones(vis.shape, dtype=np.float64)
    if sigma_n:
        weights /= sigma_n**2
    flag = np.zeros(vis.shape, dtype=np.bool)
    data_dict = {}
    data_dict["DATA"] = vis
    data_dict["MODEL_DATA"] = model_data
    data_dict["WEIGHT_SPECTRUM"] = weights
    data_dict["TIME"] = time
    data_dict["ANTENNA1"] = antenna1
    data_dict["ANTENNA2"] = antenna2
    data_dict["FLAG"] = flag
    data_dict['JONES'] = jones
    return data_dict


@corr_shape_parametrization
def test_corrupt_vis(corr_shape, jones_shape, mode):
    """
    Tests corrupt vis against predict_vis. They should do
    the same thing but corrupt_vis adheres to the structure
    in the africanus.calibration module.
    """
    np.random.seed(42)
    # simulate noise free data with random DDE's
    n_dir = 3
    n_time = 32
    n_chan = 16
    n_ant = 7
    sigma_n = 0.0
    sigma_f = 0.05
    data_dict = make_data(sigma_n, sigma_f, n_time, n_chan,
                          n_ant, n_dir, corr_shape, jones_shape, mode)
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
def test_residual_vis(corr_shape, jones_shape, mode):
    """
    Tests subtraction of model by subtracting all but one
    direction from noise free simulated data and comparing
    the output to the unsubtracted direction.
    """
    np.random.seed(42)
    # simulate noise free data with random DDE's
    n_dir = 3
    n_time = 32
    n_chan = 16
    n_ant = 7
    n_bl = n_ant*(n_ant-1)//2
    n_row = n_bl*n_time
    sigma_n = 0.0
    sigma_f = 0.05
    data_dict = make_data(sigma_n, sigma_f, n_time, n_chan,
                          n_ant, n_dir, corr_shape, jones_shape, mode)
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
                            flag, model_subtract, mode)
    # apply gains to the unsubtracted direction
    vis_unsubtracted = np.zeros((n_row, n_chan) + corr_shape,
                                dtype=np.complex128)
    vis_unsubtracted = corrupt_vis(time_bin_indices,
                                   time_bin_counts,
                                   ant1, ant2,
                                   jones_unsubtracted,
                                   vis_unsubtracted,
                                   model_unsubtracted,
                                   mode)
    # residual should now be equal to unsubtracted vis
    assert_array_almost_equal(residual, vis_unsubtracted, decimal=10)


@corr_shape_parametrization
def test_correct_vis(corr_shape, jones_shape, mode):
    """
    Tests correct_vis by correcting noise free simulation
    with random DIE gains
    """
    np.random.seed(42)
    # simulate noise free data with only DIE's
    n_dir = 1
    n_time = 32
    n_chan = 16
    n_ant = 7
    sigma_n = 0.0
    sigma_f = 0.05
    data_dict = make_data(sigma_n, sigma_f, n_time,
                          n_chan, n_ant, n_dir, corr_shape,
                          jones_shape, mode)
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
        ant1, ant2, jones, vis, flag, mode)
    # squeeze out dir axis to get expected model data
    model = model.squeeze()
    assert_array_almost_equal(corrected_vis, model, decimal=10)

@corr_shape_parametrization
def test_corrupt_vis_dask(corr_shape, jones_shape, mode):
    np.random.seed(42)
    # simulate noise free data with only DIE's
    n_dir = 3
    n_time = 32
    n_chan = 16
    n_ant = 7
    sigma_n = 0.0
    sigma_f = 0.05
    data_dict = make_data(sigma_n, sigma_f, n_time,
                          n_chan, n_ant, n_dir, corr_shape,
                          jones_shape, mode)
    vis = data_dict['DATA']  # what we need to compare to
    ant1 = data_dict['ANTENNA1']
    ant2 = data_dict['ANTENNA2']
    model = data_dict['MODEL_DATA']
    jones = data_dict['JONES']
    time = data_dict['TIME']

    # get chunking scheme
    ncpu = 8
    utimes_per_chunk = n_time//ncpu
    row_chunks = chunkify_rows(time, utimes_per_chunk)

    # set up dask arrays
    _, time_bin_idx, _, time_bin_counts = unique_time(time)
    # print(time_bin_idx)
    # print(time_bin_counts)
    da_time_bin_idx = da.from_array(time_bin_idx, chunks=(utimes_per_chunk))
    da_time_bin_counts = da.from_array(time_bin_counts, chunks=(utimes_per_chunk))
    # print(da_time_bin_idx)
    # print(da_time_bin_counts)
    da_ant1 = da.from_array(ant1, chunks=row_chunks)
    da_ant2 = da.from_array(ant2, chunks=row_chunks)
    da_model = da.from_array(model, chunks=(row_chunks, (n_chan,), (n_dir,), corr_shape))
    da_jones = da.from_array(jones, chunks=(utimes_per_chunk, n_ant, n_chan, n_dir)+jones_shape)
    vis2 = np.zeros_like(vis, dtype=vis.dtype)
    da_vis = da.from_array(vis2, chunks=(row_chunks, (n_chan,), corr_shape))

    from africanus.calibration.dask import corrupt_vis
    da_vis = corrupt_vis(da_time_bin_idx, da_time_bin_counts, da_ant1, da_ant2, da_jones, da_vis, da_model, mode)

    vis2 = da_vis.compute()
    import matplotlib.pyplot as plt
    for i in range(2):
        plt.figure('np')
        plt.imshow(np.abs(vis[:, :, i] - vis2[:, :, i]))
        plt.colorbar()
        # plt.figure('dask')
        # plt.imshow(np.abs(vis2[:, :, i]))
        # plt.colorbar()
        plt.show()
    assert_array_almost_equal(vis, vis2, decimal=10)

corr_shape=(2,)
jones_shape=(2,)
mode=literal(0)
test_corrupt_vis_dask(corr_shape, jones_shape, mode)