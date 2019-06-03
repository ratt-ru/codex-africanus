# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
np.random.seed(42)
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest
from africanus.dft import im_to_vis
from africanus.averaging.support import unique_time
from africanus.calibration import phase_only_GN

n_time = 32
n_chan = 16
n_dir = 3
n_ant = 7
n_bl = n_ant*(n_ant-1)//2
n_row = n_bl*n_time
n_cor = 2

@pytest.fixture
def give_lm():
    l = 0.1*np.random.randn(n_dir)
    m = 0.1*np.random.randn(n_dir)
    lm = np.vstack((l, m)).T
    return l, m, lm

@pytest.fixture
def give_flux():
    return 0.1 + np.random.random(n_dir)


def make_dual_pol_data(sigma_n):
    # make aux data
    antenna1 = np.zeros(n_row, dtype=np.int16)
    antenna2 = np.zeros(n_row, dtype=np.int16)
    times = np.zeros(n_row, dtype=np.float32)
    uvw = np.zeros((n_row, 3), dtype=np.float32)
    time = np.linspace(0,1,n_time)
    freq = np.linspace(1e9,2e9,n_chan)
    for i in range(n_time):
        row = 0
        for p in range(n_ant):
            for q in range(p):
                times[i*n_bl + row] = time[i]
                antenna1[i*n_bl + row] = p
                antenna2[i*n_bl + row] = q
                uvw[i*n_bl + row] = np.random.randn(3)
                row += 1
    # simulate visibilities
    model_data = np.zeros((n_dir, n_row, n_chan, n_cor), dtype=np.complex64)
    # make up some sources
    l, m, lm = give_lm()
    flux = give_flux()
    # simulate the model data
    for dir in range(n_dir):
        this_lm = lm[dir].reshape(1, 2)
        this_flux = np.tile(flux[dir], (n_chan)).reshape(1, n_chan)
        model_tmp = im_to_vis(this_flux, uvw, this_lm, freq)
        model_data[dir, :, :, 0] = model_tmp
        model_data[dir, :, :, 1] = model_tmp 
    # get corresponding noisy data
    vis = np.sum(model_data, axis=0) + sigma_n*(np.random.randn(n_row, n_chan, n_cor) +\
           1.0j*np.random.randn(n_row, n_chan, n_cor)) 
    weights = np.ones((n_row, n_chan, n_cor), dtype=np.float32)
    if sigma_n:
        weights /= sigma_n**2
    flag = np.zeros((n_row, n_chan, n_cor), dtype=np.bool)
    data_dict = {}
    data_dict["DATA"] = vis
    data_dict["MODEL_DATA"] = model_data
    data_dict["WEIGHT_SPECTRUM"] = weights
    data_dict["TIME"] = times
    data_dict["ANTENNA1"] = antenna1
    data_dict["ANTENNA2"] = antenna2
    data_dict["FLAG"] = flag
    return data_dict

def test_phase_only_diag_diag():
    """
    Calibrate ideal observation (no noise, no gains)
    and ensure that we return unity gains
    """
    # get data
    data_dict = make_dual_pol_data(0.0)
    # get time indices and counts
    _, time_inices, _, counts = unique_time(data_dict['TIME'])
    ant1 = data_dict['ANTENNA1']
    ant2 = data_dict['ANTENNA2']
    vis = data_dict['DATA']
    model = data_dict['MODEL_DATA']
    weight = data_dict['WEIGHT_SPECTRUM']
    flag = data_dict['FLAG']
    precision = 5

    gains, jhj, jhr, k = phase_only_GN(time_inices, ant1, ant2, counts,
                                       vis, flag, model, weight, tol=10**(-precision))
    gains_true = np.ones((n_time, n_ant, n_chan, n_dir, n_cor), dtype=np.complex64)
    for p in range(n_ant):
        for q in range(p):
            for s in range(n_dir):
                for c in range(n_cor):
                    diff_true = np.angle(gains_true[:, p, :, c, c] * np.conj(gains_true[:, q, :, c, c]))
                    diff_inferred = np.angle(gains[:, p, :, s, c] * np.conj(gains[:, q, :, s, c]))
                    assert_array_almost_equal(diff_true, diff_inferred, decimal=precision-1)
    return

test_phase_only_diag_diag()