#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
np.random.seed(42)
import pyrap.tables as pt
from africanus.calibration import scalar_phase_only
import matplotlib.pyplot as plt
from africanus.rime.predict import predict_vis

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

# sample DI gains
unique_times = np.unique(time)
from africanus.gps.kernels import exponential_squared
sigmat = 0.25
lt = 0.1
n_tim = unique_times.size
# n_tim = 100
t = np.linspace(0, 1, n_tim)
Kt = exponential_squared(t, t, sigmat, lt)
sigmanu = 0.5
lnu = 0.25
n_chan = freqs.size
# n_chan = 150
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

data = data.reshape(n_row, n_chan, 4)
model = model.reshape(1, n_row, n_chan, 4)

# we will use only diagonal correlations so remove the off diagonals
I = slice(0,1)
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
gains, jac, residual = scalar_phase_only.phase_only_calibration(model, data, weight, ant1, ant2, time, flag)

# plot the gains n_tim, n_ant, n_chan, n_dir, n_cor
gains = gains.squeeze()
n_ant = np.maximum(ant1.max(), ant2.max()) + 1
for p in range(3):
    for q in range(p):
        # plt.figure(str(p)+str(q))
        # plt.imshow(np.angle(gains[:, p, :] * gains[:, q, :]))
        # plt.colorbar()
        # plt.figure(str(p)+str(q)+'true')
        # plt.imshow(np.angle(gains_true[:, p, :, 0, 0] * gains_true[:, q, :, 0, 0]))
        # plt.colorbar()
        plt.figure('diff'+str(p)+str(q))
        plt.imshow(np.angle(gains[:, p, :] * np.conj(gains[:, q, :])) - np.angle(gains_true[:, p, :, 0, 0] * np.conj(gains_true[:, q, :, 0, 0])))
        plt.colorbar()
plt.show()
