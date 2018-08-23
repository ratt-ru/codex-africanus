import numpy as np


iFs = np.fft.ifftshift
Fs = np.fft.fftshift


def F(x):
    return Fs(np.fft.fft2(iFs(x), norm='ortho'))


def iF(x):
    return Fs(np.fft.ifft2(iFs(x), norm='ortho'))


def PSF_response(image, PSF_hat):

    im_hat = F(image)

    vis = PSF_hat*im_hat

    return iF(vis).real


def PSF_adjoint(img, PSF_hat):
    """
    Perform the adjoint operator of the PSF convolution; a cross correlation
    :param vis:
    :param PSF:
    :return:
    """

    vis = F(img)

    im_pad = iF(PSF_hat.conj()*vis)

    return im_pad.real


def diag_probe(A, dim, maxiter=10000, tol=1e-6, mode="Bernoulli"):

    D = np.zeros(dim, dtype='complex128')
    t = np.zeros(dim, dtype='complex128')
    q = np.zeros(dim, dtype='complex128')

    if mode == "Normal":
        gen_random = lambda npix: np.random.randn(npix)
    elif mode == "Bernoulli":
        gen_random = lambda npix: np.where(np.random.random(npix) < 0.5, -1, 1)

    for k in range(maxiter):
        v = gen_random(dim)
        t += A(v) * v
        q += v*v
        D_new = t/q
        norm = np.linalg.norm(D_new)
        rel_norm = np.linalg.norm(D_new - D)/norm
        print("relative norm: ", rel_norm)
        if rel_norm < tol:
            "Took {0} iterations to find the diagonal".format(k)
            return D
        D = D_new
    return D
