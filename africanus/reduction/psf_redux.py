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


def make_dim_reduce_ops(operator, adjoint, vis, Sigma, npix, lm):
    """
    Generate gridded visibilities, PSF_hat for response operation and noise weights
    :param operator: the response operator to be used
    :param adjoint: the adjoint operator of the response
    :param vis: the visibility data set to be reduced
    :param Sigma: the baseline weights used for a whole bunch of stuff (mainly weighting and making PSF)
    :param npix: the number of pixels in one dimension of the output image
    :param lm: the lm coordinates of each pixel of the image for creating the exact adjoint FFT
    :return: Dimensionally reduced matrices: gridded visibilities, gridded visibility space PSF, new Sigma diagonal
    """

    # generate gridded visibilites
    im_dirty = operator(Sigma*vis).reshape(npix, npix)
    vis_grid = FFT(im_dirty)

    # generate PSF_hat for execution
    PSF = operator(Sigma).reshape(npix, npix)
    PSF_hat = FFT(PSF)

    # generate new weights and return
    Sigma_hat = make_Sigma_hat(operator, adjoint, Sigma, npix, lm)
    half_sigma = np.sqrt(1/Sigma_hat)

    return vis_grid, PSF_hat, half_sigma


def make_Sigma_hat(operator, adjoint, Sigma, npix, lm):
    """
    Make the reduced Sigma operator of length npix**2
    :param operator:
    :param adjoint:
    :param Sigma:
    :param npix:
    :return:
    """
    try:
        Sigma_hat = np.fromfile('Sigma_hat.dat', dtype='float64').reshape([npix ** 2, 1])
    except:
        print("Reduced Sigma could not be found, please wait while it is generated")

        if np.DataSource.exists(None, 'F.dat'):
            FFT_mat = np.fromfile('F.dat', dtype='complex128').reshape([npix ** 2, npix ** 2])
        else:
            delta = lm[1, 0] - lm[0, 0]
            F_norm = npix ** 2
            Ffreq = da.fft.fftshift(da.fft.fftfreq(npix, d=delta, chunks=(npix, 1)))
            jj, kk = da.meshgrid(Ffreq, Ffreq)
            j = jj.reshape([npix**2, 1])
            k = kk.reshape([npix**2, 1])
            l = lm[:, 0]
            m = lm[:, 1]
            FFT_mat = np.exp(-2j * np.pi * (j * l + k * m)) / np.sqrt(F_norm)
            FFT_mat.tofile('F.dat')

        FFTH = FFT_mat.conj().T
        FFTH_dask = da.from_array(FFTH, chunks=(npix**2, 1))
        covariance_vector = FFT(adjoint(Sigma.dot(operator(FFTH_dask))).compute())
        Sigma_hat = da.diag(covariance_vector).real.compute()

        Sigma_hat.tofile('Sigma_hat.dat')

    return Sigma_hat


def PSF_response(image, PSF_hat, Sigma):

    # separate image into channels
    im_hat = FFT(image)

    # apply element-wise product with PSF_hat for convolution
    vis = PSF_hat*im_hat

    # apply weights
    w_im = Sigma*vis.flatten()

    new_im = w_im.reshape(im_hat.shape)

    return new_im


def PSF_adjoint(vis_grid, PSF_hat, Sigma):
    """
    Perform the adjoint operator of the PSF convolution; a cross correlation
    :param vis_grid: the gridded visibilities computed as the FFT of the dirty image
    :param PSF_hat: the PSF in visibility space, looks similar to uv coverage, FFT of the PSF
    :return: new_im a cross-corellated PSF with the dirty image to produce a slightly cleaner image
    """

    w_im = (Sigma * vis_grid.flatten()).reshape(vis_grid.shape)

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

