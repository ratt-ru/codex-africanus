#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numba
import numpy as np

@numba.jit(nopython=True, nogil=True, cache=True)
def _fit_spi_components(data, weights, freqs, freq0, alphas, i0s, tol=1e-6, maxiter=100):
    ncomps, nfreqs = data.shape
    jac = np.zeros((2, nfreqs), dtype=np.float64)
    alphavars = np.zeros(ncomps, dtype=np.float64)
    i0vars = np.zeros(ncomps, dtype=np.float64)
    w = freqs/freq0
    for comp in range(ncomps):
        eps = 1.0
        k = 0
        alphak = alphas[comp]
        i0k = i0s[comp]
        while eps > tol and k < maxiter:
            alphap = alphak
            i0p = i0k
            jac[1, :] = w**alphak
            model = i0k*jac[1, :]
            jac[0, :] = model * np.log(w)
            # model, jac[0, :], jac[1, :] = evaluate_model_and_jac(w, alphak, i0k)
            residual = data[comp] - model
            lik = 0.0
            hess00 = 0.0
            hess01 = 0.0
            hess11 = 0.0
            jr0 = 0.0
            jr1 = 0.0
            for v in range(nfreqs):
                lik += residual[v] * weights[v] * residual[v]
                jr0 += jac[0, v] * weights[v] * residual[v]
                jr1 += jac[1, v] * weights[v] * residual[v]
                hess00 += jac[0, v] * weights[v] * jac[0, v]
                hess01 += jac[0, v] * weights[v] * jac[1, v]
                hess11 += jac[1, v] * weights[v] * jac[1, v]
            det = hess00 * hess11 - hess01**2
            alphak = alphap + (hess11 * jr0 - hess01 * jr1)/det
            i0k = i0p + (-hess01 * jr0 + hess00 * jr1)/det
            eps = np.maximum(np.abs(alphak - alphap), np.abs(i0k - i0p))
            k += 1
        if k >= maxiter:
            print("Warning - max iterations exceeded for component ", comp)
        alphas[comp] = alphak
        alphavars[comp] = hess11/det
        i0s[comp] = i0k
        i0vars[comp] = hess00/det
    return alphas, alphavars, i0s, i0vars


alphas = np.array([0.7], dtype=np.float64)
i0s = np.array([5.0], dtype=np.float64)
nfreqs = 100
freqs = np.linspace(0.5, 1.5, nfreqs)
freq0 = 0.7
model = np.array([i0s * (freqs/freq0) ** alphas])
sigma = 0.1
data = model + sigma * np.random.randn(nfreqs)

alphai = alphas - 1.4
i0i = i0s - 4.0
weights = np.ones(nfreqs, dtype=np.float64)/sigma**2
alpha, alphavar, i0, i0var = _fit_spi_components(data, weights, freqs, freq0, alphai, i0i)

print(alpha, np.sqrt(alphavar), i0, np.sqrt(i0var))

from scipy.optimize import curve_fit

