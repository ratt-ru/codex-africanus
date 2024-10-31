#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulates phase-only gains where phases are drawn from a
Gaussian process with covariance function given by cov_func
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pyrap.tables import table
from africanus.gps.kernels import exponential_squared as cov_func
from africanus.linalg import kronecker_tools as kt
from africanus.coordinates.coordinates import radec_to_lm
import argparse


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", type=str)
    p.add_argument("--lsm", type=str)
    p.add_argument("--gain_file", type=str)
    return p


args = create_parser().parse_args()

# get times and normalise
ms = table(args.ms)
time = ms.getcol("TIME")
ant1 = ms.getcol("ANTENNA1")
ant2 = ms.getcol("ANTENNA2")
n_ant = int(np.maximum(ant1.max(), ant2.max()) + 1)
ms.close()
time = np.unique(time)
time -= time.min()
time /= time.max()
time = np.ascontiguousarray(time)
n_time = time.size
time_cov = cov_func(time, time, 0.25, 0.2)


# get freqs and normalise
spw = table(args.ms + "::SPECTRAL_WINDOW")
freq = spw.getcol("CHAN_FREQ").squeeze()
spw.close()
freq -= freq.min()
freq /= freq.max()
freq = np.ascontiguousarray(freq)
n_freq = freq.size
freq_cov = cov_func(freq, freq, 1.0, 0.25)

# get directions
sources = np.loadtxt(args.lsm, skiprows=1, usecols=(1, 2))
lm = radec_to_lm(sources)
lm = np.ascontiguousarray(lm)
n_dir = lm.shape[0]
lm_cov = cov_func(lm, lm, 1.0, 0.5)

# create kronecker matrix and get cholesky factor
K = np.array([time_cov, freq_cov, lm_cov], dtype=object)
L = kt.kron_cholesky(K)

# sample phases
gains = np.zeros((n_time, n_ant, n_freq, n_dir, 2))
for p in range(n_ant):
    xi = np.random.randn(n_time * n_freq * n_dir)
    samp = kt.kron_matvec(L, xi).reshape(n_time, n_freq, n_dir)
    gains[:, p, :, :, 0] = samp
    xi = np.random.randn(n_time * n_freq * n_dir)
    samp = kt.kron_matvec(L, xi).reshape(n_time, n_freq, n_dir)
    gains[:, p, :, :, 1] = samp

# convert to gains
gains = np.exp(1.0j * gains)

# save result
np.save(args.gain_file, gains)
