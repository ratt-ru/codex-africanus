#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

import numpy as np
from numpy.testing import assert_array_almost_equal

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
    ncorr = 2
    image = np.zeros((npix, npix, nchan, ncorr), dtype=np.float64)
    I0 = 1.0
    ref_freq = frequency[nchan // 2]
    Inu = I0 * (frequency / ref_freq) ** (-0.7)
    for corr in range(ncorr):
        image[npix // 2, npix // 2, :, corr] = Inu
    image = image.reshape(npix**2, nchan, ncorr)

    vis = im_to_vis(image, uvw, lm, frequency)

    for chan in range(nchan):
        for corr in range(ncorr):
            tmp = vis[:, chan, corr] - Inu[chan]
            assert np.all(tmp.real < 1e-13)
            assert np.all(tmp.imag < 1e-13)


def test_im_to_vis_simple():
    """
    This test checks that the result matches the analytic result
    w = 0 for multiple channels and sources but a single correlation
    """
    from africanus.dft.kernels import im_to_vis
    from africanus.constants import minus_two_pi_over_c

    np.random.seed(123)
    nrow = 100
    uvw = np.random.random(size=(nrow, 3))
    nchan = 3
    frequency = np.linspace(1.0e9, 2.0e9, nchan, endpoint=True)
    nsource = 5
    I0 = np.random.randn(nsource)
    ref_freq = frequency[nchan // 2]
    image = I0[:, None] * (frequency / ref_freq) ** (-0.7)
    # add correlation axis
    image = image[:, :, None]
    l = 0.001 + 0.1 * np.random.random(nsource)  # noqa
    m = 0.001 + 0.1 * np.random.random(nsource)
    lm = np.vstack((l, m)).T
    vis = im_to_vis(image, uvw, lm, frequency).squeeze()

    vis_true = np.zeros([nrow, nchan], dtype=np.complex128)

    for ch in range(nchan):
        for source in range(nsource):
            l, m = lm[source]
            n = np.sqrt(1.0 - l**2 - m**2)
            phase = (
                minus_two_pi_over_c
                * frequency[ch]
                * 1.0j
                * (uvw[:, 0] * l + uvw[:, 1] * m + uvw[:, 2] * (n - 1))
            )

            vis_true[:, ch] += image[source, ch, 0] * np.exp(phase)
    assert_array_almost_equal(vis, vis_true, decimal=14)


@pytest.mark.parametrize("convention", ["fourier", "casa"])
def test_im_to_vis_fft(convention):
    """
    Test against the fft when uv on regular and w is zero.
    """
    from africanus.dft.kernels import im_to_vis

    np.random.seed(123)
    Fs = np.fft.fftshift
    iFs = np.fft.ifftshift

    # set image and take fft
    npix = 29
    ncorr = 1
    image = np.zeros((npix, npix, ncorr), dtype=np.float64)
    fft_image = np.zeros((npix, npix, ncorr), dtype=np.complex128)
    nsource = 25
    for corr in range(ncorr):
        Ix = np.random.randint(5, npix - 5, nsource)
        Iy = np.random.randint(5, npix - 5, nsource)
        image[Ix, Iy, corr] = np.random.randn(nsource)
        fft_image[:, :, corr] = Fs(np.fft.fft2(iFs(image[:, :, corr])))

    # image space coords
    deltal = 0.001
    # this assumes npix is odd
    l_coord = np.arange(-(npix // 2), npix // 2 + 1) * deltal
    ll, mm = np.meshgrid(l_coord, l_coord)
    lm = np.vstack((ll.flatten(), mm.flatten())).T
    # uv-space coords
    u = Fs(np.fft.fftfreq(npix, d=deltal))
    uu, vv = np.meshgrid(u, u)
    uvw = np.zeros((npix**2, 3), dtype=np.float64)
    uvw[:, 0] = uu.flatten()
    uvw[:, 1] = vv.flatten()
    nchan = 1
    image = image.reshape(npix**2, nchan, ncorr)
    frequency = np.ones(nchan, dtype=np.float64)
    from africanus.constants import c as lightspeed

    frequency *= lightspeed  # makes result independent of frequency

    # take DFT and compare
    vis = im_to_vis(image, uvw, lm, frequency, convention=convention)
    fft_image = fft_image.reshape(npix**2, nchan, ncorr)
    fft_image = np.conj(fft_image) if convention == "casa" else fft_image

    assert_array_almost_equal(vis, fft_image, decimal=13)


def test_adjointness():
    """
    The above tests only test im_to_vis but since vis_to_im
    is simply the adjoint (up to machine precision) that is
    all we need to test.
    """
    from africanus.dft.kernels import im_to_vis as R
    from africanus.dft.kernels import vis_to_im as RH
    from africanus.constants import c as lightspeed

    np.random.seed(123)
    nsource = 21
    nrow = 31
    nchan = 3
    ncorr = 4

    uvw = 100 * np.random.random(size=(nrow, 3))
    ll = 0.01 * np.random.randn(nsource)
    mm = 0.01 * np.random.randn(nsource)
    lm = np.vstack((ll, mm)).T
    frequency = np.arange(1, nchan + 1) * lightspeed  # avoid overflow

    shape_im = (nsource, nchan, ncorr)
    size_im = np.prod(shape_im)
    gamma_im = np.random.randn(nsource, nchan, ncorr)
    shape_vis = (nrow, nchan, ncorr)
    size_vis = np.prod(shape_vis)
    gamma_vis = np.random.randn(nrow, nchan, ncorr)
    flag = np.zeros(shape_vis, dtype=bool)

    LHS = (
        gamma_vis.reshape(size_vis, 1).T.dot(
            R(gamma_im, uvw, lm, frequency).reshape(size_vis, 1)
        )
    ).real
    RHS = (
        RH(gamma_vis, uvw, lm, frequency, flag)
        .reshape(size_im, 1)
        .T.dot(gamma_im.reshape(size_im, 1))
    ).real

    assert np.abs(LHS - RHS) < 1e-13


def test_vis_to_im_flagged():
    """
    The above doesn't apply any flags so we need to test that
    separately. Note that weights for flagged data need to be
    set to zero for operators to be self adjoint.
    """
    from africanus.dft.kernels import vis_to_im
    from africanus.constants import c as lightspeed

    np.random.seed(123)
    nsource = 21
    nrow = 31
    nchan = 3
    ncorr = 4

    uvw = 100 * np.random.random(size=(nrow, 3))
    uvw[0, :] = 0.0
    ll = 0.01 * np.random.randn(nsource)
    mm = 0.01 * np.random.randn(nsource)
    lm = np.vstack((ll, mm)).T
    frequency = np.arange(1, nchan + 1) * lightspeed  # avoid overflow

    vis = np.random.randn(nrow, nchan, ncorr) + 1.0j * np.random.randn(
        nrow, nchan, ncorr
    )
    vis[0, :, :] = 1.0

    flags = np.ones((nrow, nchan, ncorr), dtype=bool)
    flags[0, :, :] = 0

    frequency = np.ones(nchan, dtype=np.float64) * lightspeed
    im_of_vis = vis_to_im(vis, uvw, lm, frequency, flags)

    assert_array_almost_equal(
        im_of_vis, np.ones((nsource, nchan, ncorr), dtype=np.float64), decimal=13
    )


@pytest.mark.parametrize("convention", ["fourier", "casa"])
def test_im_to_vis_dask(convention):
    """
    Tests against numpy version
    """
    da = pytest.importorskip("dask.array")
    from africanus.dft.kernels import im_to_vis as np_im_to_vis
    from africanus.dft.dask import im_to_vis as dask_im_to_vis
    from africanus.constants import c as lightspeed

    nrow = 8000
    uvw = 100 * np.random.random(size=(nrow, 3))
    nsource = 800  # must be odd for this test to work
    ll = 0.01 * np.random.randn(nsource)
    mm = 0.01 * np.random.randn(nsource)
    lm = np.vstack((ll, mm)).T
    nchan = 11
    frequency = np.linspace(1.0, 2.0, nchan) * lightspeed
    ncorr = 4
    image = np.random.randn(nsource, nchan, ncorr)

    # set up dask arrays
    uvw_dask = da.from_array(uvw, chunks=(nrow // 8, 3))
    lm_dask = da.from_array(lm, chunks=(nsource, 2))
    frequency_dask = da.from_array(frequency, chunks=nchan // 2)
    image_dask = da.from_array(image, chunks=(nsource, nchan // 2, ncorr))

    vis = np_im_to_vis(image, uvw, lm, frequency, convention=convention)
    vis_dask = dask_im_to_vis(
        image_dask, uvw_dask, lm_dask, frequency_dask, convention=convention
    ).compute()

    assert_array_almost_equal(vis, vis_dask, decimal=13)


def test_vis_to_im_dask():
    """
    Tests against numpy version
    """
    da = pytest.importorskip("dask.array")
    from africanus.dft.kernels import vis_to_im as np_vis_to_im
    from africanus.dft.dask import vis_to_im as dask_vis_to_im
    from africanus.constants import c as lightspeed

    nchan = 11
    nrow = 8000
    nsource = 80
    ncorr = 4

    vis = np.random.randn(nrow, nchan, ncorr)
    uvw = np.random.randn(nrow, 3)

    ll = 0.01 * np.random.randn(nsource)
    mm = 0.01 * np.random.randn(nsource)
    lm = np.vstack((ll, mm)).T
    nchan = 11
    frequency = np.linspace(1.0, 2.0, nchan) * lightspeed

    flagged_frac = 0.45
    flags = np.random.choice(
        a=[False, True], size=(nrow, nchan, ncorr), p=[flagged_frac, 1 - flagged_frac]
    )

    image = np_vis_to_im(vis, uvw, lm, frequency, flags)

    # set up dask arrays
    uvw_dask = da.from_array(uvw, chunks=(nrow // 8, 3))
    lm_dask = da.from_array(lm, chunks=(nsource, 2))
    frequency_dask = da.from_array(frequency, chunks=nchan // 2)
    vis_dask = da.from_array(vis, chunks=(nrow // 8, nchan // 2, ncorr))
    flags_dask = da.from_array(flags, chunks=(nrow // 8, nchan // 2, ncorr))

    image_dask = dask_vis_to_im(
        vis_dask, uvw_dask, lm_dask, frequency_dask, flags_dask
    ).compute()

    assert_array_almost_equal(image, image_dask, decimal=13)


def test_symmetric_covariance():
    """
    Test that the image plane precision matrix R^H Sigma^-1R is Hermitian
    (symmetric since its real).
    """
    from africanus.dft.kernels import vis_to_im, im_to_vis

    np.random.seed(123)

    nsource = 25
    ncorr = 1
    nchan = 1

    lmmax = 0.05
    ll = -lmmax + 2 * lmmax * np.random.random(nsource)  # noqa
    mm = -lmmax + 2 * lmmax * np.random.random(nsource)
    lm = np.vstack((ll, mm)).T

    nrows = 1000
    uvw = np.random.randn(nrows, 3) * 1000
    uvw[:, 2] = 0.0

    freq = np.array([1.0e9])

    flags = np.zeros((nrows, nchan, ncorr), dtype=np.bool_)

    # get the "psf" matrix at source locations
    psf_source = np.zeros((nsource, nsource), dtype=np.float64)
    point_source = np.ones((1, nchan, ncorr), dtype=np.float64)
    for source in range(nsource):
        lm_i = lm[source].reshape(1, 2)
        Ki = im_to_vis(point_source, uvw, lm_i, freq)
        psf_source[:, source] = vis_to_im(Ki, uvw, lm, freq, flags).squeeze()

    assert_array_almost_equal(psf_source, psf_source.T, decimal=14)
