import numpy as np


iFs = np.fft.ifftshift
Fs = np.fft.fftshift


def F(x):
    return Fs(np.fft.fft2(iFs(x), norm='ortho'))


def iF(x):
    return Fs(np.fft.ifft2(iFs(x), norm='ortho'))


def PSF_response(image, PSF_hat, Sigma):

    # im_pad = np.pad(image, padding, 'constant')
    im_hat = F(image)

    # apply element-wise product with PSF_hat for convolution
    vis = PSF_hat*im_hat

    # apply weights
    w_im = Sigma*vis.flatten()

    new_im = iF(w_im.reshape(im_hat.shape))

    return new_im


def PSF_adjoint(image, PSF_hat, Sigma):
    """
    Perform the adjoint operator of the PSF convolution; a cross correlation
    :param vis:
    :param PSF:
    :return:
    """

    im_hat = F(image).flatten()

    w_im = (Sigma * im_hat).reshape(image.shape)

    vis = PSF_hat.conj()*w_im

    new_im = iF(vis)

    return new_im


def diag_probe(A, dim, maxiter=2000, tol=1e-8, mode="Bernoulli"):

    D = np.zeros(dim**2, dtype='complex128')
    t = np.zeros(dim**2, dtype='complex128')
    q = np.zeros(dim**2, dtype='complex128')

    if mode == "Normal":
        gen_random = lambda npix: np.random.randn(npix).reshape([dim, dim])
    elif mode == "Bernoulli":
        gen_random = lambda npix: np.where(np.random.random(npix) < 0.5, -1, 1)

    for k in range(maxiter):
        v = gen_random(dim**2)
        t += A(v.reshape([dim, dim])).flatten()
        q += v*v
        D_new = t/q
        norm = np.linalg.norm(D_new)
        rel_norm = np.linalg.norm(D_new - D)/norm
        if k % 10 == 0:
            print("relative norm {0}: {1}".format(k, rel_norm))
        if rel_norm < tol:
            "Took {0} iterations to find the diagonal".format(k)
            return D
        D = D_new
    print("Final relative norm: ", rel_norm)
    return D
