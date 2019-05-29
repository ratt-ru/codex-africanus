#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
np.random.seed(42)
import pyrap.tables as pt
from africanus.calibration import phase_only_GN
import matplotlib.pyplot as plt
from africanus.rime.predict import predict_vis
from numpy.testing import assert_array_equal, assert_array_almost_equal

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms")
    return p

args = create_parser().parse_args()

# get the data
ms = pt.table(args.ms)

weight = ms.getcol('WEIGHT')
ant1 = ms.getcol("ANTENNA1")
ant2 = ms.getcol("ANTENNA2")
time = ms.getcol('TIME')
# data = ms.getcol('CLEAN_DATA')  # this will be produced by sampling and applying gains
model = ms.getcol('MODEL_DATA')
flag = ms.getcol('FLAG')
ms.close()

freqtab = pt.table(args.ms+"/SPECTRAL_WINDOW")
freqs = freqtab.getcol("CHAN_FREQ").squeeze()
freqtab.close()

# sample DI diagonal gains
unique_times = np.unique(time)
from africanus.gps.kernels import exponential_squared
sigmat = 0.25
lt = 0.1
n_tim = unique_times.size
t = np.linspace(0, 1, n_tim)
Kt = exponential_squared(t, t, sigmat, lt)
sigmanu = 0.5
lnu = 0.25
n_chan = freqs.size
nu = np.linspace(0, 1, n_chan)
Knu = exponential_squared(nu, nu, sigmanu, lnu)
K = np.array([Kt, Knu])
n_ant = np.maximum(ant1.max(), ant2.max()) + 1
n_cor = 2
gains_true = np.zeros((n_tim, n_ant, n_chan, n_cor, n_cor), dtype=np.complex128)

import africanus.linalg.kronecker_tools as kt
L = kt.kron_cholesky(K)
n_tot = n_tim * n_chan
for p in range(n_ant):
    for c in range(n_cor):
        xi = np.random.randn(n_tot)
        samp = kt.kron_matvec(L, xi).reshape(n_tim, n_chan)
        # plt.figure()
        # plt.imshow(samp)
        # plt.colorbar()
        # plt.show()
        gains_true[:, p, :, c, c] = np.exp(1.0j*samp)

# apply gains
tind = np.unique(time, return_inverse=True)[1]
n_row, _, _ = model.shape

# add extra dimension for direction axis and reshape correlation axis into 2x2
model = model[None].reshape(1, n_row, n_chan, 2, 2)
data = predict_vis(tind, ant1, ant2, source_coh=model, die1_jones=gains_true, die2_jones=gains_true)

# we will use only diagonal correlations so remove the off diagonals
I = ((0,0), (1,1))
data = data[:, :, I]
model = model[:, :, :, I]

flag = flag[:, :, I]
if len(weight.shape)==3:
    weight = weight[:, :, I]
elif len(weight.shape)==2:
    weight = weight[:, None, I]
    print("No weight spectrum supplied. Duplicating weights for all frequencies")
else:
    raise Exception("Incorrect shape for weights")

# do calibration
precision = 8
maxiter = 100
gains, jhj, jhr, k = phase_only_GN(model, data, weight, ant1, ant2, \
                                                     time, flag, tol=10**(-precision), maxiter=maxiter)

if k > maxiter:
    print("Warning - maximum iterations exceeded")

# plot the gains n_tim, n_ant, n_chan, n_dir, n_cor
n_ant = np.maximum(ant1.max(), ant2.max()) + 1
n_dir = 1
for p in range(n_ant):
    for q in range(p):
        for s in range(n_dir):
            for c in range(n_cor):
                diff_true = np.angle(gains_true[:, p, :, c, c] * np.conj(gains_true[:, q, :, c, c]))
                diff_inferred = np.angle(gains[:, p, :, s, c] * np.conj(gains[:, q, :, s, c]))
                assert_array_almost_equal(diff_true, diff_inferred, decimal=precision-1)
