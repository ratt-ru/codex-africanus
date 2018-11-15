# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

import numpy as np
import dask.array as da

import pytest


def test_diagonal_probe():
    """
    Test that the diagonal probe correctly approximates a complex matrix's diagonal
    """
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
    """
    Test that the matrix diagonal can be correctly approximated, this method is more accurate but slower.
    """
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
    """
    Test the adjointness of the PSF response, checks that when operating on the same matrices, the product of
    the operators produce the same result
    """
    from africanus.reduction.psf_redux import PSF_adjoint, PSF_response, FFT
    # test the self adjointness of the PSF operator
    pix = 10
    PSF = np.random.random([pix, pix])
    PSF_hat = PSF_hat = FFT(PSF)
    wsum = pix**2

    P = lambda image: PSF_response(image, PSF_hat) * np.sqrt(pix ** 2 / wsum)
    PH = lambda image: PSF_adjoint(image, PSF_hat) * np.sqrt(pix ** 2 / wsum)

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

    vis = PSF_response(image, PSF_hat)/wsum

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

    vis = PSF_response(image, PSF_hat)

    vis_true = PSF_hat*FFT(image)

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

    vis_psf = PSF_response(image, PSF_hat)/wsum

    assert abs(vis.real - vis_psf.real) < 1e-12


def test_FFT():
    """
    Here we test that the FFT operator equates to an analytic solution of the same image.
    """
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
    offset_l = 5
    offset_m = 1
    image[npix // 2 + offset_l, npix // 2 + offset_m] = 1

    # fourier transform image
    im_test = FFT(image) * npix

    # analytically fourier transform it
    delta = lm[1, 0] - lm[0, 0]
    im_fft = np.zeros(npix ** 2, dtype='complex128')
    Ffreq = np.fft.fftshift(np.fft.fftfreq(npix, d=delta))
    jj, kk = np.meshgrid(Ffreq, Ffreq)

    im_fft = np.exp(-2j * np.pi * (kk*delta*offset_l + jj*delta*offset_m))

    # compare
    assert np.all(abs(im_test - im_fft) < 1e-14)


def test_fft_dft_single_centre():
    """
    We test that the DFT and FFT of a same sized image with centred source produce the same result: a matrix of 1 + 0j
    """
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


def test_compare_response():
    """We compare the results of FR^H\SigmaR produces the same result as F(PSF)*F(), this can only be observed in
    image space, so both results are acted on by iFFT before comparing"""
    from africanus.dft.kernels import vis_to_im, im_to_vis
    from africanus.reduction.psf_redux import FFT, iFFT, PSF_response
    from africanus.opts.data_reader import gen_image_space, gen_padding_space

    # setup parameters
    np.random.seed(111)
    nrow = 1000
    uvw = np.random.randn(nrow, 3)*2
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
    for i in range(nsource):
        pos = np.random.randint(3, npix-3, size=2)
        image[pos[0], pos[1]] = np.random.randn(1)

    # pad the image
    pad_im = np.pad(image, padding, mode='constant')

    # generate PSF and PSF_hat
    PSF = vis_to_im(weight, uvw, lm_pad, frequency).reshape(pad_pix, pad_pix)
    PSF_hat = FFT(PSF)

    # get PSF transform of the image
    psf_vis = PSF_response(pad_im, PSF_hat)
    psf_reduce = iFFT(psf_vis)[padding:-padding, padding:-padding]
    psf_reduce /= abs(psf_reduce).max()

    # get DFT transfrom of the image
    im_vis = im_to_vis(image.reshape(npix ** 2, 1), uvw, lm, frequency)
    re_imaged = vis_to_im(weight*im_vis, uvw, lm_pad, frequency).reshape(pad_pix, pad_pix)
    dft_vis = FFT(re_imaged)
    dft_reduce = iFFT(dft_vis)[padding:-padding, padding:-padding]
    dft_reduce /= abs(dft_reduce).max()

    assert np.all(abs(abs(psf_reduce) - abs(dft_reduce)) < 1e-14)


def test_compare_adjoint():
    """
    Compare the results of R^H\SigmaRF^H against F^H(F(PSF).()) acting on a gridded visibility space FR^HV, this
    produces an image which can then be compared"""
    from africanus.dft.kernels import vis_to_im, im_to_vis
    from africanus.reduction.psf_redux import FFT, iFFT, PSF_adjoint
    from africanus.opts.data_reader import gen_image_space, gen_padding_space

    # setup parameters
    np.random.seed(111)
    nrow = 100
    uvw = np.random.randn(nrow, 3) * 10
    uvw[:, 2] = 0.0
    frequency = np.array([1.5e9])

    l = np.array([.02])
    m = np.array([.02])
    lm, npix, cs, fov = gen_image_space(uvw, frequency, l, m)
    lm_pad, pad_pix, padding = gen_padding_space(npix, 1, cs, fov)

    # generate random visibilities
    vis = np.random.randn(nrow, 1) + 1.0j*np.random.randn(nrow, 1)
    im_dirty = vis_to_im(vis, uvw, lm, frequency).reshape(npix, npix)
    im_pad = np.pad(im_dirty, padding, 'constant')
    vis_grid = FFT(im_pad)

    # generate PSF and PSF_hat
    weight = np.ones((nrow, 1))
    PSF = vis_to_im(weight, uvw, lm_pad, frequency).reshape(pad_pix, pad_pix)
    PSF_hat = FFT(PSF)

    # Perform adjoint operation of the DFT
    im_grid = iFFT(vis_grid)[padding:-padding, padding:-padding].reshape((npix**2, 1))
    vis_of_vis = im_to_vis(im_grid, uvw, lm, frequency)
    dft_reduce = vis_to_im(weight*vis_of_vis, uvw, lm, frequency).flatten()
    dft_reduce /= abs(dft_reduce).max()

    # perform adjoint operation of the PSF
    psf_reduce = PSF_adjoint(vis_grid, PSF_hat)[padding:-padding, padding:-padding].flatten()
    psf_reduce /= abs(psf_reduce).max()

    # compare
    assert np.all(abs(abs(psf_reduce) - abs(dft_reduce)) < 1e-14)


def test_compare_explicit_FFT():
    """
    compare the explicit effects of the FFT matrix with that of the FFT operator
    """
    from africanus.dft.kernels import vis_to_im
    from africanus.opts.data_reader import gen_image_space, gen_padding_space
    from africanus.reduction.psf_redux import FFT, make_FFT_matrix

    # setup parameters
    np.random.seed(111)
    nrow = 1000
    uvw = np.random.randn(nrow, 3)*2
    weight = np.ones((nrow, 1))
    uvw[:, 2] = 0.0
    frequency = np.array([1.5e9])
    nsource = 5

    l = np.array([.02])
    m = np.array([.02])
    lm, npix, cs, fov = gen_image_space(uvw, frequency, l, m)
    lm_pad, pad_pix, padding = gen_padding_space(npix, 1, cs, fov)

    # create or read explicit FFT matrix
    try:
        FFT_mat = np.fromfile('F.dat', dtype='complex128').reshape([pad_pix**2, pad_pix**2])
    except:
        FFT_mat = make_FFT_matrix(da.from_array(lm_pad, chunks=[npix**2, 1]), pad_pix)

    # generate semi-random image
    image = np.zeros((npix, npix))
    for i in range(nsource):
        pos = np.random.randint(3, npix-3, size=2)
        image[pos[0], pos[1]] = np.random.randn(1)

    # pad the image
    pad_im = np.pad(image, padding, mode='constant')

    # generate PSF and PSF_hat
    PSF = vis_to_im(weight, uvw, lm, frequency).reshape(npix, npix)
    PSF_pad = np.pad(PSF, padding, 'constant')
    PSF_hat = FFT(PSF_pad)
    PSF_hat_mat = FFT_mat.dot(PSF_pad.flatten())

    # get PSF transform of the image
    im_hat = FFT(pad_im)
    psf_reduce = PSF_hat * im_hat
    psf_reduce /= abs(psf_reduce).max()

    im_hat_mat = FFT_mat.dot(pad_im.flatten())
    psf_reduce_mat = PSF_hat_mat * im_hat_mat
    psf_reduce_mat /= abs(psf_reduce_mat).max()

    assert np.all(abs(abs(psf_reduce).flatten() - abs(psf_reduce_mat)) < 1e-14)


def test_compare_explicit_DFT():
    """
    Compare the effects of an explicit DFT matrix with that of the DFT operators
    """
    from africanus.dft.kernels import vis_to_im, im_to_vis
    from africanus.reduction.psf_redux import FFT, make_DFT_matrix
    from africanus.opts.data_reader import gen_image_space, gen_padding_space

    # setup parameters
    np.random.seed(111)
    nrow = 1000
    uvw = np.random.randn(nrow, 3)*2
    weight = np.ones((nrow, 1))
    uvw[:, 2] = 0.0
    frequency = np.array([1.5e9])
    nsource = 5

    l = np.array([.02])
    m = np.array([.02])
    lm, npix, cs, fov = gen_image_space(uvw, frequency, l, m)
    lm_pad, pad_pix, padding = gen_padding_space(npix, 1, cs, fov)

    # generate dft matrix
    try:
        DFT_mat = np.fromfile('R.dat', dtype='complex128').reshape([nrow, pad_pix**2])
    except:
        DFT_mat = make_DFT_matrix(da.from_array(uvw, chunks=[nrow, 1]), da.from_array(lm_pad, chunks=[npix**2, 1]), frequency, nrow, pad_pix)

    # generate semi-random image
    image = np.zeros((npix, npix))
    for i in range(nsource):
        pos = np.random.randint(3, npix - 3, size=2)
        image[pos[0], pos[1]] = np.random.randn(1)

    # pad the image
    im_pad = np.pad(image, padding, mode='constant')

    # get DFT transfrom of the image
    im_vis = im_to_vis(im_pad.reshape(pad_pix ** 2, 1), uvw, lm_pad, frequency)
    vis_weight = weight*im_vis
    re_imaged = vis_to_im(vis_weight, uvw, lm_pad, frequency).real.reshape(pad_pix, pad_pix)
    dft_reduce = FFT(re_imaged.real)
    dft_reduce /= abs(dft_reduce).max()

    # get DFT transfrom of the image
    im_vis_mat = DFT_mat.dot(im_pad.reshape(pad_pix ** 2, 1))
    vis_weight_mat = weight*im_vis_mat
    re_imaged_mat = DFT_mat.conj().T.dot(vis_weight_mat).real.reshape(pad_pix, pad_pix)
    dft_reduce_mat = FFT(re_imaged_mat)
    dft_reduce_mat /= abs(dft_reduce_mat).max()

    assert np.all(abs(abs(im_vis_mat) - abs(im_vis)) < 1e-14)
    assert np.all(abs(abs(dft_reduce_mat) - abs(dft_reduce)) < 1e-14)
    assert np.all(abs(abs(re_imaged_mat) - abs(re_imaged)) < 1e-11)


def test_compare_diagonals():
    """
    Compare the diagonal of the FR^H\SigmaRF^H matrix with that of the PSF response
    """
    from africanus.opts.data_reader import gen_padding_space, gen_image_space
    from africanus.dft.dask import vis_to_im, im_to_vis
    from africanus.reduction.psf_redux import FFT, make_Sigma_hat, make_DFT_matrix, make_FFT_matrix

    # setup parameters
    np.random.seed(111)
    nrow = 1000
    nchunk = 8
    uvw = np.random.randn(nrow, 3) * 2
    uvw[:, 2] = 0.0
    frequency = np.array([1.5e9])

    l = np.array([.02])
    m = np.array([.02])
    lm, npix, cs, fov = gen_image_space(uvw, frequency, l, m)
    lm_pad, pad_pix, padding = gen_padding_space(npix, 1, cs, fov)

    # daskify for speed
    weight = da.eye(nrow, chunks=nrow // nchunk)
    uvw = da.from_array(uvw, chunks=(nrow // nchunk, 3))
    lm_pad = da.from_array(lm_pad, chunks=(pad_pix**2, 2))

    # read or generate FFT and DFT reduction matrix diagonal
    try:
        FFT_mat = np.fromfile('F.dat', dtype='complex128').reshape([pad_pix**2, pad_pix**2])
    except:
        FFT_mat = make_FFT_matrix(da.from_array(lm_pad, chunks=[npix**2, 1]), pad_pix)
    FFTH_mat = FFT_mat.conj().T

    DFT_mat = np.fromfile('R.dat', dtype='complex128').reshape([nrow, pad_pix ** 2])
    DFTH_mat = DFT_mat.conj().T

    DFT_response_matrix = FFT_mat.dot(DFTH_mat.dot(weight.dot(DFT_mat.dot(FFTH_mat))))
    DFT_response_matrix.tofile('Redux_response.dat')

    true_diag = np.diagonal(DFT_response_matrix)
    true_diag = true_diag/abs(true_diag).max()

    # diagonal from operators
    operator = lambda im: im_to_vis(im, uvw, lm_pad, frequency)
    adjoint = lambda vis: vis_to_im(vis, uvw, lm_pad, frequency)
    op_diag = make_Sigma_hat(operator, adjoint, weight, pad_pix, lm_pad).flatten()
    op_diag = op_diag/abs(op_diag).max()

    # generate PSF hat
    PSF = adjoint(np.diagonal(weight.compute()).reshape([nrow, 1]))
    PSF_hat = FFT(PSF).flatten()
    PSF_hat = PSF_hat/abs(PSF_hat).max()

    assert np.all(abs(true_diag - op_diag) < 1e14)


if __name__=="__main__":
    test_compare_explicit_DFT()
