# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

import numpy as np
import pytest


def test_fit_spi_components_vs_scipy():
    """
    Here we just test the per component spi fitter against
    a looped version of scipy's curve_fit
    :return:
    """
    from africanus.model.spi import fit_spi_components
    curve_fit = pytest.importorskip("scipy.optimize").curve_fit

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
    alpha1, alphavar1, I01, I0var1 = fit_spi_components(
        data, weights, freqs.squeeze(), freq0, tol=1e-8)

    def spi_func(nu, I0, alpha):
        return I0 * nu ** alpha

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

    np.testing.assert_array_almost_equal(alpha1, alpha2, decimal=6)
    # note variances not necessarily accurate to within tol because
    # scipy uses LM instead of GN
    np.testing.assert_array_almost_equal(alphavar1, alphavar2, decimal=3)
    np.testing.assert_array_almost_equal(I01, I02, decimal=6)
    np.testing.assert_array_almost_equal(I0var1, I0var2, decimal=3)


def test_dask_fit_spi_components_vs_np():
    from africanus.model.spi import fit_spi_components as np_fit_spi
    from africanus.model.spi.dask import fit_spi_components
    da = pytest.importorskip("dask.array")

    np.random.seed(123)

    ncomps = 800
    alphas = -0.7 + 0.25 * np.random.randn(ncomps, 1)
    i0s = 5.0 + np.random.randn(ncomps, 1)
    nfreqs = 1000
    freqs = np.linspace(0.5, 1.5, nfreqs).reshape(1, nfreqs)
    freq0 = 0.7
    model = i0s * (freqs / freq0) ** alphas
    sigma = np.abs(0.25 + 0.1 * np.random.randn(nfreqs))
    data = model + sigma[None, :] * np.random.randn(ncomps, nfreqs)

    weights = 1.0/sigma**2
    freqs = freqs.squeeze()
    alpha1, alphavar1, I01, I0var1 = np_fit_spi(data, weights, freqs, freq0)

    # now for the dask version
    data_dask = da.from_array(data, chunks=(100, nfreqs))
    weights_dask = da.from_array(weights, chunks=(nfreqs))
    freqs_dask = da.from_array(freqs, chunks=(nfreqs))

    alpha2, alphavar2, I02, I0var2 = fit_spi_components(data_dask,
                                                        weights_dask,
                                                        freqs_dask,
                                                        freq0).compute()

    np.testing.assert_array_almost_equal(alpha1, alpha2, decimal=6)
    np.testing.assert_array_almost_equal(alphavar1, alphavar2, decimal=6)
    np.testing.assert_array_almost_equal(I01, I02, decimal=6)
    np.testing.assert_array_almost_equal(I0var1, I0var2, decimal=6)
