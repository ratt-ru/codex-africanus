# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.testing import assert_array_almost_equal
from africanus.averaging.support import unique_time
from africanus.calibration import phase_only_gauss_newton
from africanus.calibration.tests.test_utils import make_data
from numba.types.misc import literal


def test_phase_only_diag_diag():
    """
    Test phase only calibration by checking that
    we reconstruct the correct gains for a noise
    free simulation.
    """
    np.random.seed(42)
    # simulate noise free data with random DDE's
    n_dir = 1
    n_time = 32
    n_chan = 16
    n_ant = 7
    sigma_n = 0.0
    sigma_f = 0.05
    corr_shape = (2,)
    jones_shape = (2,)
    mode = literal(0)
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
    weight = data_dict['WEIGHT_SPECTRUM']
    # calibrate the data
    jones0 = np.ones((n_time, n_ant, n_chan, n_dir) + jones_shape,
                     dtype=np.complex64)
    precision = 5
    gains, jhj, jhr, k = phase_only_gauss_newton(
        time_bin_indices, time_bin_counts,
        ant1, ant2, jones0, vis,
        flag, model, weight,
        tol=10**(-precision), maxiter=100)
    # check that phase differences are correct
    for p in range(n_ant):
        for q in range(p):
            phase_diff_true = np.angle(jones[:, p] * np.conj(jones[:, q]))
            phase_diff = np.angle(gains[:, p] * np.conj(gains[:, q]))
            assert_array_almost_equal(
                phase_diff_true, phase_diff, decimal=precision-1)
