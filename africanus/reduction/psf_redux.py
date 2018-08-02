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

    return vis


def PSF_adjoint(vis, PSF_hat):
    """
    Perform the adjoint operator of the PSF convolution; a cross correlation
    :param vis:
    :param PSF:
    :param pad_size:
    :return:
    """

    im_pad = iF(PSF_hat.conj()*vis)

    return im_pad
