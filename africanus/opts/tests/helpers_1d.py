import dask.array as da
import scipy as sp
import scipy.fftpack as scifft
import numpy as np

c = 2.99792458e8
two_pi_over_c = 2*sp.pi/c


def FFT(x):
    return scifft.fftshift(scifft.fft(scifft.ifftshift(x)))


def iFFT(x):
    return scifft.fftshift(scifft.ifft(scifft.ifftshift(x)))


def im_to_vis(image, uvw, lm):
    # For each uvw coordinate
    vis_of_im = np.zeros_like(uvw, dtype=np.complex128)
    for row in range(uvw.shape[0]):
        u = uvw[row]

        # For each source
        for source in range(lm.shape[0]):
            l = lm[source]
            vis_of_im[row] += sp.exp(-2.0j * np.pi * u * l)*image[source]

    return vis_of_im


def vis_to_im(vis, u_row, l_row):
    # For each source
    im_of_vis = np.zeros_like(l_row, dtype=np.complex128)
    for source in range(l_row.shape[0]):
        l_point = l_row[source]

        # for each u coord
        for row in range(u_row.shape[0]):
            u = u_row[row]
            im_of_vis[source] += (sp.exp(2.0j * np.pi * u * l_point) * vis[row])

    return im_of_vis


def make_fft_matrix(l_sources, npix):

    # create spacing for fft range
    delta = l_sources[1] - l_sources[0]

    # create normalisation constant
    F_norm = npix

    # generate fft frequency range
    Ffreq = np.fft.fftshift(np.fft.fftfreq(npix, d=delta))
    Ffreq = Ffreq.reshape([npix, 1])

    # actually calculate the fft matrix
    FFT_mat = np.exp(-2j * np.pi * Ffreq.dot(l_sources.T))/np.sqrt(F_norm)

    return FFT_mat


def make_dft_matrix(uvw, lm, weights=None):
    if weights is None:
        wsum = uvw.size
    else:
        wsum = np.sum(weights)

    DFT_mat = np.exp(-2.0j * np.pi * (uvw.dot(lm.T)))

    return DFT_mat
