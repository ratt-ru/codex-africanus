# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

import numpy as np
import dask.array as da

import pytest


def test_diagonal_probe():
    from africanus.reduction.psf_redux import diag_probe
    # first construct a positive semi-definite operator
    Asqrt = np.random.randn(25, 10) + 1.0j * np.random.randn(25, 10)
    A = Asqrt.conj().T.dot(Asqrt)

    # get true diagonal
    diag_true = np.diag(A)

    # now get the diagonal via probing
    Aop = lambda x: A.dot(x)
    diag_estimate = diag_probe(Aop, 10)
    diff = abs(diag_true - diag_estimate)
    print(diff)
    assert np.all(diff < 1e-12)


def test_guess_matrix():
    from africanus.reduction.psf_redux import guess_matrix
    # again create that positive semi-definte op
    Asqrt = np.random.randn(25, 10) + 1.0j * np.random.randn(25, 10)
    A = Asqrt.conj().T.dot(Asqrt)

    # get true diagonal
    diag_true = np.diag(A)

    # now get the diagonal via probing
    Aop = lambda x: A.dot(x)
    diag_estimate = guess_matrix(Aop, 10)
    diff = abs(diag_true - diag_estimate)
    assert np.all(diff < 1e-12)


def test_PSF_adjointness():
    from africanus.reduction.psf_redux import PSF_adjoint, PSF_response, F
    # test the self adjointness of the PSF operator
    pix = 10
    PSF = np.random.random([pix, pix])
    PSF_hat = PSF_hat = F(PSF)
    sigma = np.ones(pix ** 2)
    wsum = pix**2

    P = lambda image: PSF_response(image, PSF_hat, sigma) * np.sqrt(pix ** 2 / wsum)
    PH = lambda image: PSF_adjoint(image, PSF_hat, sigma) * np.sqrt(pix ** 2 / wsum)

    gamma1 = np.random.randn(pix ** 2).reshape([pix, pix])
    gamma2 = np.random.randn(pix ** 2).reshape([pix, pix])

    LHS = gamma2.flatten().T.dot(P(gamma1).flatten()).real
    RHS = PH(gamma2).flatten().T.dot(gamma1.flatten()).real

    assert np.abs(LHS - RHS) < 1e-12


def test_PSF_phase_centre():
    """
    The simplest test here is to see if a single source at the phase centre
    returns simply the flux everywhere with zero imaginary part
    """
    from africanus.dft.kernels import vis_to_im
    from africanus.reduction.psf_redux import PSF_response, F, sigma_approx

    nrow = 100
    uvw = np.random.random(size=(nrow, 3))
    weights = np.ones((nrow, 1))
    npix = 35  # must be odd for this test to work
    x = np.linspace(-0.1, 0.1, npix)
    ll, mm = np.meshgrid(x, x)
    lm = np.vstack((ll.flatten(), mm.flatten())).T
    nchan = 1
    frequency = np.linspace(1.0, 2.0, nchan, endpoint=True)

    image = np.zeros([npix, npix, nchan], dtype=np.float64)
    I0 = 1.0
    ref_freq = frequency[nchan//2]
    Inu = I0*(frequency/ref_freq)**(-0.7)
    image[npix//2, npix//2, :] = Inu
    image = image.reshape(npix**2, nchan)

    wsum = sum(weights)
    PSF = vis_to_im(weights, uvw, lm, frequency)/np.sqrt(wsum)
    PSF_hat = F(PSF)
    Sigma = sigma_approx(PSF)

    vis = PSF_response(image, PSF_hat, Sigma)/wsum

    for i in range(nchan):
        tmp = vis[:, i] - Inu[i]
        assert np.all(tmp.real < 1e-13)
        assert np.all(tmp.imag < 1e-13)


def test_PSF_zero_w():
    """
    This test checks that the result matches the analytic result
    in the case when w = 0 for multiple channels and sources.
    """
    from africanus.dft.kernels import vis_to_im
    from africanus.reduction.psf_redux import PSF_response, sigma_approx, F, iF

    np.random.seed(123)
    nrow = 100
    uvw = np.random.random(size=(nrow, 3))
    weight = np.ones((nrow, 1))
    uvw[:, 2] = 0.0
    nchan = 1
    frequency = np.linspace(1.e9, 2.e9, nchan, endpoint=True)
    nsource = 5

    I0 = np.random.randn(nsource)
    ref_freq = frequency[nchan//2]
    image = I0[:, None] * (frequency/ref_freq)**(-0.7)
    l = 0.001 + 0.1*np.random.random(nsource)
    m = 0.001 + 0.1*np.random.random(nsource)
    lm = np.vstack((l, m)).T

    wsum = sum(weight)
    PSF = vis_to_im(weight, uvw, lm, frequency)/np.sqrt(wsum)
    PSF_hat = F(PSF)
    Sigma = sigma_approx(PSF)

    vis = PSF_response(image, PSF_hat, Sigma)

    vis_true = iF((Sigma*(PSF_hat*F(image)).flatten()).reshape(PSF.shape))

    assert np.allclose(vis, vis_true)


def test_PSF_single():
    """
    Here we check that the result is consistent for
    a single baseline, source and channel.
    """
    from africanus.reduction.psf_redux import PSF_response, F, sigma_approx
    from africanus.dft.kernels import vis_to_im, im_to_vis

    nrow = 1
    uvw = np.random.random(size=(nrow, 3))
    weights = np.ones((nrow, 1))
    frequency = np.array([1.5e9])
    l = 0.015
    m = -0.0123
    lm = np.array([[l, m]]) # np.vstack(([l], [m])).T
    image = np.array([[1.0]])
    vis = im_to_vis(image, uvw, lm, frequency)
    print(vis)

    PSF = vis_to_im(weights, uvw, lm, frequency)/np.sqrt(sum(weights))
    PSF_hat = F(PSF)
    sigma = sigma_approx(PSF)

    vis_psf = PSF_response(image, PSF_hat, sigma)/sum(weights)

    assert np.allclose(vis, vis_psf)
