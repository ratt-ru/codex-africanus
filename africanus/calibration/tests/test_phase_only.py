# -*- coding: utf-8 -*-


import numpy as np
from numpy.testing import assert_array_almost_equal
from africanus.dft import im_to_vis
from africanus.averaging.support import unique_time
from africanus.calibration import phase_only_gauss_newton
from africanus.rime.predict import predict_vis

n_time = 32
n_chan = 16
n_ant = 7
n_bl = n_ant*(n_ant-1)//2
n_row = n_bl*n_time
n_cor = 2


def give_lm(n_dir):
    ls = 0.1*np.random.randn(n_dir)
    ms = 0.1*np.random.randn(n_dir)
    lm = np.vstack((ls, ms)).T
    return lm


def give_flux(n_dir):
    return 0.1 + np.random.random(n_dir)


def make_dual_pol_data(sigma_n, n_dir, sigma_f):
    # make aux data
    antenna1 = np.zeros(n_row, dtype=np.int16)
    antenna2 = np.zeros(n_row, dtype=np.int16)
    time = np.zeros(n_row, dtype=np.float32)
    uvw = np.zeros((n_row, 3), dtype=np.float32)
    unique_time = np.linspace(0, 1, n_time)
    freq = np.linspace(1e9, 2e9, n_chan)
    for i in range(n_time):
        row = 0
        for p in range(n_ant):
            for q in range(p):
                time[i*n_bl + row] = unique_time[i]
                antenna1[i*n_bl + row] = p
                antenna2[i*n_bl + row] = q
                uvw[i*n_bl + row] = np.random.randn(3)
                row += 1
    assert time.size == n_row
    # simulate visibilities
    model_data = np.zeros((n_row, n_chan, n_dir, n_cor), dtype=np.complex64)
    # make up some sources
    lm = give_lm(n_dir)
    flux = give_flux(n_dir)
    # simulate model data (pure Stokes I)
    for dir in range(n_dir):
        this_lm = lm[dir].reshape(1, 2)
        this_flux = np.tile(flux[dir], (n_chan, n_cor))[None, :, :]
        model_tmp = im_to_vis(this_flux, uvw, this_lm, freq)
        model_data[:, :, dir, :] = model_tmp
    assert not np.isnan(model_data).any()
    # simulate gains (just radnomly scattered around 1 for now)
    jones = np.ones((n_time, n_ant, n_chan, n_dir, n_cor), dtype=np.complex64)
    if sigma_f:
        jones += sigma_f*(
            np.random.randn(n_time, n_ant, n_chan, n_dir, n_cor) +
            1.0j*np.random.randn(n_time, n_ant, n_chan, n_dir, n_cor))
        assert (np.abs(jones) > 1e-5).all()
        assert not np.isnan(jones).any()
    # get vis
    time_index = np.unique(time, return_inverse=True)[1]
    jones_tmp = np.transpose(jones, [3, 0, 1, 2, 4])
    model_tmp = np.transpose(model_data, [2, 0, 1, 3])
    vis = predict_vis(
        time_index, antenna1, antenna2,
        source_coh=model_tmp,
        dde1_jones=jones_tmp,
        dde2_jones=jones_tmp)
    assert not np.isnan(vis).any()
    # add noise
    if sigma_n:
        vis += sigma_n*(np.random.randn(n_row, n_chan, n_cor) +
                        1.0j*np.random.randn(n_row, n_chan, n_cor))
    weights = np.ones((n_row, n_chan, n_cor), dtype=np.float32)
    if sigma_n:
        weights /= sigma_n**2
    flag = np.zeros((n_row, n_chan, n_cor), dtype=np.bool)
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


def test_phase_only_diag_diag():
    """
    Test phase only calibration by checking that
    we reconstruct the correct gains for a noise
    free simulation.
    """
    np.random.seed(42)
    # simulate noise free data with random DDE's
    n_dir = 1
    sigma_n = 0.0
    sigma_f = 0.05
    data_dict = make_dual_pol_data(sigma_n, n_dir, sigma_f)
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
    jones0 = np.ones((n_time, n_ant, n_chan, n_dir, n_cor), dtype=np.complex64)
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
