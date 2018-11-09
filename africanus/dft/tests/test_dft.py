#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

import numpy as np

import pytest


def test_im_to_vis_phase_centre():
    """
    The simplest test here is to see if a single source at the phase centre
    returns simply the flux everywhere with zero imaginary part
    """
    from africanus.dft.kernels import im_to_vis

    nrow = 100
    uvw = np.random.random(size=(nrow, 3))
    npix = 35  # must be odd for this test to work
    x = np.linspace(-0.1, 0.1, npix)
    ll, mm = np.meshgrid(x, x)
    lm = np.vstack((ll.flatten(), mm.flatten())).T
    nchan = 11
    frequency = np.linspace(1.0, 2.0, nchan, endpoint=True)

    image = np.zeros([npix, npix, nchan], dtype=np.float64)
    I0 = 1.0
    ref_freq = frequency[nchan//2]
    Inu = I0*(frequency/ref_freq)**(-0.7)
    image[npix//2, npix//2, :] = Inu
    image = image.reshape(npix**2, nchan)

    vis = im_to_vis(image, uvw, lm, frequency)

    for i in range(nchan):
        tmp = vis[:, i] - Inu[i]
        assert np.all(tmp.real < 1e-13)
        assert np.all(tmp.imag < 1e-13)


def test_im_to_vis_zero_w():
    """
    This test checks that the result matches the analytic result
    in the case when w = 0 for multiple channels and sources.
    """
    from africanus.dft.kernels import im_to_vis
    from africanus.constants import minus_two_pi_over_c
    np.random.seed(123)
    nrow = 100
    uvw = np.random.random(size=(nrow, 3))
    uvw[:, 2] = 0.0
    nchan = 3
    frequency = np.linspace(1.e9, 2.e9, nchan, endpoint=True)
    nsource = 5
    I0 = np.random.randn(nsource)
    ref_freq = frequency[nchan//2]
    image = I0[:, None] * (frequency/ref_freq)**(-0.7)
    l = 0.001 + 0.1*np.random.random(nsource)
    m = 0.001 + 0.1*np.random.random(nsource)
    lm = np.vstack((l, m)).T
    vis = im_to_vis(image, uvw, lm, frequency)

    vis_true = np.zeros([nrow, nchan], dtype=np.complex128)

    for ch in range(nchan):
        for source in range(nsource):
            phase = (minus_two_pi_over_c*frequency[ch] * 1.0j *
                     (uvw[:, 0]*lm[source, 0] +
                      uvw[:, 1]*lm[source, 1]))

            vis_true[:, ch] += image[source, ch]*np.exp(phase)

    assert np.allclose(vis, vis_true)


def test_im_to_vis_single_baseline_and_chan():
    """
    Here we check that the result is consistent for
    a single baseline, source and channel.
    """
    from africanus.dft.kernels import im_to_vis
    from africanus.constants import minus_two_pi_over_c
    nrow = 1
    uvw = np.random.random(size=(nrow, 3))
    frequency = np.array([1.5e9])
    l = 0.015
    m = -0.0123
    lm = np.array([[l, m]])
    n = np.sqrt(1 - l**2 - m**2)
    image = np.array([[1.0]])
    vis = im_to_vis(image, uvw, lm, frequency)

    vis_true = image*np.exp(minus_two_pi_over_c * frequency * 1.0j *
                            (uvw[:, 0]*l + uvw[:, 1]*m + uvw[:, 2]*(n - 1.0)))

    assert np.allclose(vis, vis_true)


def test_vis_to_im():
    """
    Still thinking of a better test here but the simplest test
    does exactly the same as the above.
    If we have an auto-correlation we expect
    to measure a flat image with value wsum
    """
    from africanus.dft.kernels import vis_to_im
    nchan = 11

    vis = np.ones([1, nchan], dtype=np.complex128)
    uvw = np.zeros([1, 3], dtype=np.float64)
    npix = 5
    x = np.linspace(-0.1, 0.1, npix)
    ll, mm = np.meshgrid(x, x)
    lm = np.vstack((ll.flatten(), mm.flatten())).T
    wsum = 1.0

    frequency = np.linspace(1.0, 2.0, nchan, endpoint=True)

    image = vis_to_im(vis, uvw, lm, frequency)

    for i in range(nchan):
        assert np.all(image[:, i] == wsum)


def test_adjointness():
    """
    She is the mother of all tests.
    The DFT should be perfectly self adjoint up to machine precision.
    """
    from africanus.dft.kernels import im_to_vis as R
    from africanus.dft.kernels import vis_to_im as RH

    np.random.seed(123)
    Npix = 33
    Nvis = 1000
    Nchan = 1

    uvw = np.random.random(size=(Nvis, 3))
    x = np.linspace(-0.1, 0.1, Npix)
    ll, mm = np.meshgrid(x, x)
    lm = np.vstack((ll.flatten(), mm.flatten())).T
    frequency = np.array([1.0])

    gamma1 = np.random.randn(Npix**2, Nchan)
    gamma2 = np.random.randn(Nvis, Nchan)

    LHS = (gamma2.T.dot(R(gamma1, uvw, lm, frequency))).real
    RHS = (RH(gamma2, uvw, lm, frequency).T.dot(gamma1)).real
    assert np.all(np.abs(LHS - RHS) < 1e-11)


def test_im_to_vis_dask():
    da = pytest.importorskip("dask.array")
    from africanus.dft.kernels import im_to_vis as np_im_to_vis
    from africanus.dft.dask import im_to_vis as dask_im_to_vis

    nrow = 100
    uvw = np.random.random(size=(nrow, 3))
    npix = 35  # must be odd for this test to work
    x = np.linspace(-0.1, 0.1, npix)
    ll, mm = np.meshgrid(x, x)
    lm = np.vstack((ll.flatten(), mm.flatten())).T
    nchan = 11
    frequency = np.linspace(1.0, 2.0, nchan, endpoint=True)

    image = np.zeros([npix, npix, nchan], dtype=np.float64)
    I0 = 1.0
    ref_freq = frequency[nchan//2]
    Inu = I0*(frequency/ref_freq)**(-0.7)
    image[npix//2, npix//2, :] = Inu
    image = image.reshape(npix**2, nchan)

    # set up dask arrays
    uvw_dask = da.from_array(uvw, chunks=(25, 3))
    lm_dask = da.from_array(lm, chunks=(npix**2, 2))
    frequency_dask = da.from_array(frequency, chunks=4)
    image_dask = da.from_array(image, chunks=(npix**2, 4))

    vis = np_im_to_vis(image, uvw, lm, frequency)
    vis_dask = dask_im_to_vis(image_dask, uvw_dask,
                              lm_dask, frequency_dask).compute()

    assert np.allclose(vis, vis_dask)


def test_vis_to_im_dask():
    da = pytest.importorskip("dask.array")
    from africanus.dft.kernels import vis_to_im as np_vis_to_im
    from africanus.dft.dask import vis_to_im as dask_vis_to_im

    nchan = 11

    vis = np.ones([100, nchan], dtype=np.complex128)
    uvw = np.zeros([100, 3], dtype=np.float64)
    npix = 5
    x = np.linspace(-0.1, 0.1, npix)
    ll, mm = np.meshgrid(x, x)
    lm = np.vstack((ll.flatten(), mm.flatten())).T

    frequency = np.linspace(1.0, 2.0, nchan, endpoint=True)

    image = np_vis_to_im(vis, uvw, lm, frequency)

    # set up dask arrays
    uvw_dask = da.from_array(uvw, chunks=(25, 3))
    lm_dask = da.from_array(lm, chunks=(5, 2))
    frequency_dask = da.from_array(frequency, chunks=4)
    vis_dask = da.from_array(vis, chunks=(25, 4))

    image_dask = dask_vis_to_im(
        vis_dask, uvw_dask, lm_dask, frequency_dask).compute()

    assert np.allclose(image, image_dask)


def test_symmetric_covariance():
    """
    Test that the image plane covariance matrix R^H\Sigma^-1R is Hermitian
    (symmetric since its real).
    """
    from africanus.dft.kernels import vis_to_im
    from africanus.constants.consts import minus_two_pi_over_c
    np.random.seed(123)

    lmmax = 0.05
    nsource = 25

    l = -0.8*lmmax + 1.6*lmmax*np.random.random(nsource)
    m = -0.8 * lmmax + 1.6 * lmmax * np.random.random(nsource)
    lm = np.vstack((l, m)).T

    nrows = 1000
    uvw = np.random.randn(nrows, 3) * 1000
    uvw[:, 2] = 0.0

    freq = np.array([1.0e9])

    # get the "psf" matrix at source locations
    psf_source = np.zeros((nsource, nsource), dtype=np.float64)
    for source in range(nsource):
        l, m = lm[source]
        n = np.sqrt(1 - l ** 2 - m ** 2)
        Ki = np.zeros([nrows, 1], dtype=np.complex128)
        for row in range(nrows):
            Ki[row] = np.exp(1j*minus_two_pi_over_c*freq[0] *
                             (uvw[row, 0]*l + uvw[row, 1]*m) +
                             uvw[row, 2]*(n - 1))
        psf_source[:, source:source+1] = vis_to_im(Ki, uvw, lm, freq)

    assert np.allclose(psf_source, psf_source.T, atol=1e-12, rtol=1e-10)
