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
    im_dirty = adjoint(Sigma*vis).reshape(npix, npix)
    vis_grid = FFT(im_dirty)

    # generate PSF_hat for execution
    PSF = adjoint(Sigma).reshape(npix, npix)
    PSF_hat = FFT(PSF)

    # generate new weights and return
    Sigma_hat = make_Sigma_hat(operator, adjoint, Sigma, npix, lm)

    return vis_grid, PSF_hat, Sigma_hat


def make_Sigma_hat(operator, adjoint, sigma, npix, lm):
    """
    Make the reduced Sigma operator of length npix**2
    :param operator:
    :param adjoint:
    :param sigma:
    :param npix:
    :return:
    """
    try:
        sigma_hat = np.fromfile('Sigma_hat.dat', dtype='float64').reshape([npix ** 2, 1])
    except:
        print("Reduced Sigma could not be found, please wait while it is generated")

        try:
            FFT_mat = np.fromfile('F.dat', dtype='complex128').reshape([npix ** 2, npix ** 2])
        except:
            FFT_mat = make_FFT_matrix(lm, npix)

        FFTH = FFT_mat.conj().T
        FFTH_dask = da.from_array(FFTH, chunks=(npix**2, 1))
        covariance_vector = FFT(adjoint(sigma.dot(operator(FFTH_dask))).compute())
        sigma_hat = da.diag(covariance_vector).real.compute()

        sigma_hat.tofile('Sigma_hat.dat')

    return sigma_hat


def make_FFT_matrix(lm, npix):
    """
    Generates an FFT matrix basically just for FRRF matrix
    :param lm: the spacial range over which the FFT must be calculated
    :param npix: the number of pixels in lm
    :return: FFT_mat, an FFT matrix of size npix**2 x npix**2
    """

    # create spacing for fft range
    delta = lm[1, 0] - lm[0, 0]

    # create normalisation constant
    F_norm = npix ** 2

    # generate fft frequency range
    Ffreq = da.fft.fftshift(da.fft.fftfreq(npix, d=delta, chunks=(npix)))

    # create vectors to be dotted together to create FFT matrix (hopefully) more efficiently with dask than a loop
    jj, kk = da.meshgrid(Ffreq, Ffreq)
    j = da.rechunk(jj.reshape([npix ** 2, 1]), chunks=[npix ** 2 // 10, 1])
    k = da.rechunk(kk.reshape([npix ** 2, 1]), chunks=[npix ** 2 // 10, 1])
    l = da.rechunk(lm[:, 0].reshape([1, npix ** 2]), chunks=[1, npix ** 2])
    m = da.rechunk(lm[:, 1].reshape([1, npix ** 2]), chunks=[1, npix ** 2])

    # actually calculate the fft matrix
    FFT_mat = (da.exp(-2j * np.pi * (j.dot(l) + k.dot(m))) / da.sqrt(F_norm)).compute()

    # store the matrix to file so we don't have to go through this too often
    FFT_mat.tofile('F.dat')

    return FFT_mat


def make_DFT_matrix(uvw, lm, frequency, nrow, npix):
    from africanus.constants.consts import minus_two_pi_over_c

    u = da.rechunk(uvw[:, 0].reshape([nrow, 1]), chunks=[nrow//10, 1])
    v = da.rechunk(uvw[:, 1].reshape([nrow, 1]), chunks=[nrow // 10, 1])
    w = da.rechunk(uvw[:, 2].reshape([nrow, 1]), chunks=[nrow // 10, 1])

    l = da.rechunk(lm[:, 0].reshape([1, npix ** 2]), chunks=[1, npix ** 2])
    m = da.rechunk(lm[:, 1].reshape([1, npix ** 2]), chunks=[1, npix ** 2])
    n = da.sqrt(1.0 - l ** 2 - m ** 2) - 1.0

    DFT_mat = da.exp(1.0j * frequency[0] * minus_two_pi_over_c * (u.dot(l) + v.dot(m) + w.dot(n))).compute()
    DFT_mat.tofile('R.dat')

    return DFT_mat


def whiten_noise(grid_vis, psf_hat, sigma_hat):
    half_sigma = np.sqrt(1 / sigma_hat)

    white_vis = half_sigma*grid_vis
    white_psf_hat = half_sigma*psf_hat

    return white_vis, white_psf_hat


def PSF_response(image, PSF_hat):

    # separate image into channels
    im_hat = FFT(image)

    # apply element-wise product with PSF_hat for convolution
    vis = PSF_hat*im_hat

    return vis


def PSF_adjoint(vis_grid, PSF_hat):
    """
    Perform the adjoint operator of the PSF convolution; a cross correlation
    :param vis_grid: the gridded visibilities computed as the FFT of the dirty image
    :param PSF_hat: the PSF in visibility space, looks similar to uv coverage, FFT of the PSF
    :return: new_im a cross-corellated PSF with the dirty image to produce a slightly cleaner image
    """

    vis = PSF_hat.conj()*vis_grid

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

