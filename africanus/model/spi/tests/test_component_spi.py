#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

import numpy as np
import pytest

def test_vs_scipy():
    """
    Here we just test the per component spi fitter against
    a looped version of scipy's curve_fit
    :return: 
    """
    from africanus.model.spi import component_spi
    from scipy.optimize import curve_fit

    np.random.seed(123)

    ncomps = 25
    alphas = -0.7 + 0.25 * np.random.randn(ncomps, 1)
    i0s = 5.0 + np.random.randn(ncomps, 1)
    nfreqs = 100
    freqs = np.linspace(0.5, 1.5, nfreqs).reshape(1, nfreqs)
    freq0 = 0.7
    model = i0s * (freqs / freq0) ** alphas
    sigma = np.abs(0.25 + 0.1 * np.random.randn(nfreqs))
    data = model + sigma[None, :] * np.random.randn(ncomps, nfreqs)

    weights = 1.0/sigma**2
    alpha1, alphavar1, I01, I0var1 = component_spi.fit_spi_components(data, weights, freqs.squeeze(), freq0, tol=1e-8)

    spi_func = lambda nu, I0, alpha: I0 * nu ** alpha

    I02 = np.zeros(ncomps)
    I0var2 = np.zeros(ncomps)
    alpha2 = np.zeros(ncomps)
    alphavar2 = np.zeros(ncomps)

    for i in range(ncomps):
        popt, pcov = curve_fit(spi_func, (freqs / freq0).squeeze(), data[i, :],
                               sigma=np.diag(sigma**2),
                               p0=np.array([1.0, -0.7]))
        I02[i] = popt[0]
        I0var2[i] = pcov[0, 0]
        alpha2[i] = popt[1]
        alphavar2[i] = pcov[1, 1]


    assert np.allclose(alpha1, alpha2, atol=1e-6)
    # note variances not necessarily accurate to within tol because
    # scipy uses LM instead of GN
    assert np.allclose(alphavar1, alphavar2, atol=1e-3)
    assert np.allclose(I01, I02, atol=1e-6)
    assert np.allclose(I0var1, I0var2, atol=1e-3)


test_vs_scipy()
