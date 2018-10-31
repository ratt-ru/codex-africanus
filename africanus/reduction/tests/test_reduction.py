# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

import numpy as np
import dask.array as da

import pytest


def test_diagonal_probe():
    from africanus.reduction.psf_redux import diag_probe
    # first construct a positive semi-definite operator
    np.random.seed(111)
    npix = 10
    Asqrt = np.random.randn(npix, npix) + 1.0j * np.random.randn(npix, npix)
    A = Asqrt.conj().T.dot(Asqrt)
    A /= A.max()

    # get true diagonal
    diag_true = np.diag(A)

    # now get the diagonal via probing
    Aop = lambda x: A.dot(x)
    diag_estimate = diag_probe(Aop, npix)
    assert np.all(abs(diag_true - diag_estimate) < 1e-12)


def test_guess_matrix():
    from africanus.reduction.psf_redux import guess_matrix
    # again create that positive semi-definte op
    npix = 10
    Asqrt = np.random.randn(npix, npix) + 1.0j * np.random.randn(npix, npix)
    A = Asqrt.conj().T.dot(Asqrt)

    # get true diagonal
    diag_true = np.diag(A)

    # now get the diagonal via probing
    Aop = lambda x: A.dot(x)
    diag_estimate = guess_matrix(Aop, npix)
    diff = abs(diag_true - diag_estimate)
    assert np.all(diff < 1e-12)


def test_PSF_adjointness():
    from africanus.reduction.psf_redux import PSF_adjoint, PSF_response, FFT
    # test the self adjointness of the PSF operator
    pix = 10
    PSF = np.random.random([pix, pix])
    PSF_hat = PSF_hat = FFT(PSF)
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
    from africanus.reduction.psf_redux import PSF_response, FFT, sigma_approx

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
    PSF_hat = FFT(PSF)
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
    from africanus.reduction.psf_redux import PSF_response, sigma_approx, FFT, iFFT

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
    PSF_hat = FFT(PSF)
    Sigma = sigma_approx(PSF)

    vis = PSF_response(image, PSF_hat, Sigma)

    vis_true = iFFT((Sigma*(PSF_hat*FFT(image)).flatten()).reshape(PSF.shape))

    assert np.allclose(vis, vis_true)


def test_PSF_single():
    """
    Here we check that the result is consistent for
    a single baseline, source and channel.
    """
    from africanus.reduction.psf_redux import PSF_response, FFT
    from africanus.dft.kernels import vis_to_im, im_to_vis

    nrow = 1
    uvw = np.random.random(size=(nrow, 3))
    weights = np.ones((nrow, 1))
    frequency = np.array([1.5e9])
    l = 0.015
    m = -0.0123
    lm = np.array([[l, m]])
    image = np.array([[1.0]])
    vis = im_to_vis(image, uvw, lm, frequency)

    wsum = sum(weights)
    PSF = vis_to_im(weights, uvw, lm, frequency)/np.sqrt(wsum)
    PSF_hat = FFT(PSF)
    sigma = np.ones(PSF.shape[0])

    vis_psf = PSF_response(image, PSF_hat, sigma)/wsum

    assert abs(vis.real - vis_psf.real) < 1e-12


def test_FFT():
    from africanus.reduction.psf_redux import FFT
    from africanus.opts.data_reader import gen_image_space

    # setup parameters
    np.random.seed(111)
    nrow = 1000
    uvw = np.random.random(size=(nrow, 3)) * 10
    uvw[:, 2] = 0.0
    frequency = np.array([1.5e9])

    l = np.array([.02])
    m = np.array([.02])
    lm, npix, cs, fov = gen_image_space(uvw, frequency, l, m)

    # generate off-centre source image
    image = np.zeros((npix, npix))
    image[npix // 2 + 1, npix // 2] = 1

    # fourier transform image
    im_test = FFT(image) * npix

    # analytically fourier transform it
    delta = lm[1, 0] - lm[0, 0]
    im_fft = np.zeros(npix ** 2, dtype='complex128')
    Ffreq = np.fft.fftshift(np.fft.fftfreq(npix, d=delta))
    jj, kk = np.meshgrid(Ffreq, Ffreq)
    jk = np.vstack((jj.flatten(), kk.flatten())).T
    c = 2.99792458e8
    for h in range(npix ** 2):
        j, k = jk[h]
        im_fft[h] = np.exp(-2j * np.pi * (k * delta))

    # compare
    assert np.all(abs(im_test.flatten() - im_fft) < 1e-14)


def test_fft_dft_single_centre():
    from africanus.dft.kernels import im_to_vis
    from africanus.reduction.psf_redux import FFT
    from africanus.opts.data_reader import gen_image_space

    # setup parameters
    nrow = 29**2
    uvw = np.random.random(size=(nrow, 3)) * 10
    uvw[:, 2] = 0.0
    frequency = np.array([1.5e9])

    l = np.array([.02])
    m = np.array([.02])
    lm, npix, cs, fov = gen_image_space(uvw, frequency, l, m)

    # generate centred source
    image = np.zeros((npix, npix))
    image[npix // 2, npix // 2] = 1

    # generate both transforms over the same space
    im_fft = FFT(image)*npix
    im_dft = im_to_vis(image.reshape([npix**2, 1]), uvw, lm, frequency).reshape([npix, npix])

    assert np.all(abs(im_fft - im_dft) < 1e-14)


def test_against_dft():
    from africanus.dft.kernels import vis_to_im, im_to_vis
    from africanus.reduction.psf_redux import FFT, iFFT, sigma_approx
    from africanus.opts.data_reader import gen_image_space, plot, gen_padding_space
    import matplotlib.pyplot as plt

    # setup parameters
    np.random.seed(111)
    nrow = 1000
    uvw = np.random.randn(nrow, 3)*10
    weight = np.ones((nrow, 1))
    uvw[:, 2] = 0.0
    frequency = np.array([1.5e9])
    nsource = 5

    l = np.array([.02])
    m = np.array([.02])
    lm, npix, cs, fov = gen_image_space(uvw, frequency, l, m)
    lm_pad, pad_pix, padding = gen_padding_space(npix, 1, cs, fov)

    # generate semi-random image
    image = np.zeros((npix, npix))
    # for i in range(nsource):
    #     pos = np.random.randint(3, npix-3, size=2)
    #     image[pos[0], pos[1]] = np.random.randn(1)

    image[npix//2+1, npix//2] = 1

    # pad the image
    pad_im = np.pad(image, padding, mode='constant')

    wsum = sum(weight)

    # get PSF transform of the image
    PSF = vis_to_im(weight, uvw, lm, frequency).real.reshape(npix, npix)
    PSF_pad = np.pad(PSF, padding, 'constant')
    PSF_hat = FFT(PSF_pad)
    im_hat = FFT(pad_im)
    psf_reduce = PSF_hat*im_hat
    psf_reduce /= psf_reduce.max()

    # get DFT transfrom of the image
    im_vis = im_to_vis(image.reshape(npix ** 2, 1), uvw, lm, frequency)
    re_imaged = vis_to_im(weight*im_vis, uvw, lm, frequency).real.reshape(npix, npix)
    dft_reduce = FFT(np.pad(re_imaged, padding, mode='constant'))
    dft_reduce /= dft_reduce.max()

    # plot differences
    plot(psf_reduce.real, "PSF", pad_pix)
    plot(dft_reduce.real, "DFT", pad_pix)
    plot(psf_reduce.imag, "PSF imag", pad_pix)
    plot(dft_reduce.imag, "DFT imag", pad_pix)
    plot(abs(psf_reduce.real - dft_reduce.real), "Diff real", pad_pix)
    plot(abs(psf_reduce.imag - dft_reduce.imag), "Diff imag", pad_pix)
    plt.show()

    assert np.all(abs(psf_reduce.real - dft_reduce.real) < 1e-14)
    assert np.all(abs(psf_reduce.imag - dft_reduce.imag) < 1e-14)
