import numpy as np


iFs = np.fft.ifftshift
Fs = np.fft.fftshift

def PSF_response(image, PSF, pad_size):

    PSF_pad = np.pad(PSF, pad_size, mode='constant')
    PSF_hat = Fs(np.fft.fft2(iFs(PSF_pad)))

    # im_pad = np.pad(image, pad_size, mode='constant')
    im_hat = Fs(np.fft.fft2(iFs(image)))

    vis = PSF_hat*im_hat

    return vis  # .real[pad_size[0]:-pad_size[0], pad_size[1]:-pad_size[1]]


def PSF_adjoint(vis, PSF, pad_size):
    """
    Perform the adjoint operator of the PSF convolution; a cross correlation
    :param vis:
    :param PSF:
    :param pad_size:
    :return:
    """

    # sigma = np.diag(weights.flatten())

    # vis_pad = np.pad(vis, pad_size, mode='constant')
    # vis_hat = Fs(np.fft.ifft2(iFs(vis)))
    # vis_conj = vis.conj()

    PSF_pad = np.pad(PSF, pad_size, mode='constant')
    PSF_hat = Fs(np.fft.fft2(iFs(PSF_pad)))

    im_pad = Fs(np.fft.ifft2(iFs(PSF_hat*vis)))

    return im_pad  # [pad_size[0]:-pad_size[0], pad_size[1]:-pad_size[1]]
