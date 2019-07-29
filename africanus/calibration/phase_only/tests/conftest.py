# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from africanus.calibration.utils import corrupt_vis
from africanus.dft import im_to_vis
from africanus.averaging.support import unique_time
import pytest


def lm_factory(n_dir, rs):
    ls = 0.1*rs.randn(n_dir)
    ms = 0.1*rs.randn(n_dir)
    lm = np.vstack((ls, ms)).T
    return lm


def flux_factory(n_dir, n_chan, corr_shape, alpha, freq, freq0, rs):
    w = freq/freq0
    flux = np.zeros((n_dir, n_chan) + corr_shape, dtype=np.float64)
    for d in range(n_dir):
        tmp_flux = np.abs(rs.normal(size=corr_shape))
        for v in range(n_chan):
            flux[d, v] = tmp_flux * w[v]**alpha
    return flux


@pytest.fixture
def data_factory():
    def impl(sigma_n, sigma_f, n_time, n_chan, n_ant,
             n_dir, corr_shape, jones_shape, phase_only_gains=False):
        rs = np.random.RandomState(42)
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
        lm = lm_factory(n_dir, rs)
        alpha = -0.7
        freq0 = freq[n_chan//2]
        flux = flux_factory(n_dir, n_chan, corr_shape, alpha, freq, freq0, rs)
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
            if phase_only_gains:
                jones = np.exp(1.0j*rs.normal(loc=0.0, scale=sigma_f,
                                              size=jones.shape))
            else:
                jones += (rs.normal(loc=0.0, scale=sigma_f,
                                    size=jones.shape) +
                          1.0j*rs.normal(loc=0.0, scale=sigma_f,
                                         size=jones.shape))
            assert (np.abs(jones) > 1e-5).all()
            assert not np.isnan(jones).any()
        # get vis
        _, time_bin_indices, _, time_bin_counts = unique_time(time)
        vis = corrupt_vis(time_bin_indices, time_bin_counts,
                          antenna1, antenna2, jones, model_data)
        assert not np.isnan(vis).any()
        # add noise
        if sigma_n:
            vis += (rs.normal(loc=0.0, scale=sigma_n, size=vis.shape) +
                    1.0j*rs.normal(loc=0.0, scale=sigma_n, size=vis.shape))
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
    return impl
