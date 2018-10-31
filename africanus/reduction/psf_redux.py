import numpy as np
import dask.array as da
import numba


def FFT_dask(x):
    return da.fft.fftshift(da.fft.fft2(da.fft.ifftshift(x)))


def iFFT_dask(x):
    return da.fft.fftshift(da.fft.ifft2(da.fft.ifftshift(x)))


def FFT(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x), norm='ortho'))


def iFFT(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x), norm='ortho'))


def PSF_response(image, PSF_hat, Sigma):

    # separate image into channels
    im_hat = FFT(image)

    # apply element-wise product with PSF_hat for convolution
    vis = PSF_hat*im_hat

    # apply weights
    w_im = Sigma*vis.flatten()

    new_im = w_im.reshape(im_hat.shape)

    return iFFT(new_im)


def PSF_adjoint(image, PSF_hat, Sigma):
    """
    Perform the adjoint operator of the PSF convolution; a cross correlation
    :param vis:
    :param PSF:
    :return:
    """

    im_hat = FFT(image).flatten()

    w_im = (Sigma * im_hat).reshape(image.shape)

    vis = PSF_hat.conj()*w_im

    new_im = iFFT(vis)

    return new_im


def sigma_approx(PSF):
    psf_hat = FFT(PSF)

    P = lambda x: iFFT(psf_hat*FFT(x.reshape(PSF.shape))).flatten()

    return guess_matrix(P, PSF.size).real


def diag_probe(A, dim, maxiter=2000, tol=1e-12, mode="Bernoulli"):

    D = np.zeros(dim, dtype='complex128')
    t = np.zeros(dim, dtype='complex128')
    q = np.zeros(dim, dtype='complex128')

    if mode == "Normal":
        gen_random = lambda npix: np.random.randn(npix) + 1.0j*np.random.randn(npix)
    elif mode == "Bernoulli":
        gen_random = lambda npix: np.where(np.random.random(npix) < 0.5, -1, 1) + \
                                 1.0j*np.where(np.random.random(npix) < 0.5, -1, 1)

    for k in range(maxiter):
        v = gen_random(dim)
        t += A(v)*v.conj()
        q += v*v.conj()
        D_new = t/q
        norm = np.linalg.norm(D_new)
        rel_norm = np.linalg.norm(D_new - D)/norm
        if k % 100 == 0:
            print("relative norm {0}: {1}".format(k, rel_norm))
        if rel_norm < tol:
            print("Took {0} iterations to find the diagonal".format(k))
            return D_new
        D = D_new
    print("Final relative norm: ", rel_norm)
    return D_new


def guess_matrix(operator, N):
    '''
    Compute the covariance matrix by applying a given operator (F*Phi^T*Phi) on different delta functions
    '''

    operdiag = np.zeros(N, dtype='complex')
    for i in np.arange(N):
        deltacol = np.zeros((N, 1))
        deltacol[i] = 1.0
        deltacol = da.from_array(deltacol, chunks=(N, 1))
        currcol = operator(deltacol).flatten()
        operdiag[i] = currcol[i]

    return operdiag

    # from scipy.sparse import coo_matrix
    # from scipy.sparse import csc_matrix

    # if diagonly:
    #     maxnonzeros = min(M, N)
    #     operdiag = np.zeros(maxnonzeros, dtype='complex')
    # else:
    #     #         matrix = np.zeros((M, N))               # HUGE
    #     matrix = csc_matrix((M, N))  # economic storage
    #
    # for i in np.arange(N):
    #     deltacol = coo_matrix(([1], ([i], [0])), shape=(N, 1))
    #     currcol = operator(deltacol.toarray()).flatten()
    #     if diagonly:
    #         if i > maxnonzeros: break
    #         operdiag[i] = currcol[i]
    #     else:
    #         matrix[:, i] = currcol[:, np.newaxis]
    #
    # if diagonly:
    #     matrix = coo_matrix((operdiag, (np.arange(maxnonzeros), np.arange(maxnonzeros))), shape=(M, N))
    #
    # return matrix

