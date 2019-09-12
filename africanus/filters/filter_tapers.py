# -*- coding: utf-8 -*-


import numpy as np

from .kaiser_bessel_filter import (kaiser_bessel_with_sinc,
                                   estimate_kaiser_bessel_beta)


def taper(filter_type, ny, nx, conv_filter, **kwargs):
    r"""
    Parameters
    ----------
    filter_type : {"kaiser-bessel"}
        Type of filter
    ny : int
        Number of pixels in the v dimension.
    nx : int
        Number of pixels in the u dimension.
    conv_filter : :class:`africanus.filters.ConvolutionFilter`
        Associated Convolution Filter.

    Returns
    -------
    :class:`numpy.ndarray`
        Taper of shape :code:`(ny, nx)`
    """
    cf = conv_filter

    if filter_type == "sinc":
        return np.ones((ny, nx))
    elif filter_type == "kaiser-bessel":
        try:
            beta = kwargs.pop('beta')
        except KeyError:
            beta = estimate_kaiser_bessel_beta(cf.full_sup)

        # What would Andre Offringa do?
        # He would compute the numeric solution
        taps = np.arange(cf.no_taps) / cf.oversample - cf.full_sup // 2
        kb = kaiser_bessel_with_sinc(taps, cf.full_sup, cf.oversample, beta)
        kbshift = np.fft.fftshift(kb)

        width = nx * cf.oversample
        height = ny * cf.oversample

        # Put the first and last halves of the shifted Kaiser Bessel
        # at each end of the output buffer, then FFT
        buf = np.zeros(width, dtype=kb.dtype)
        buf[:kbshift.size // 2] = kbshift[:kbshift.size // 2]
        buf[-kbshift.size // 2:] = kbshift[-kbshift.size // 2:]
        x = np.fft.ifft(buf).real

        buf = np.zeros(height, dtype=kb.dtype)
        buf[:kbshift.size // 2] = kbshift[:kbshift.size // 2]
        buf[-kbshift.size // 2:] = kbshift[-kbshift.size // 2:]
        y = np.fft.ifft(buf).real

        # First quarter of the taper
        quarter = y[:ny // 2, None] * x[None, :nx // 2]

        # Create the taper by copying
        # the quarter into the appropriate bits
        taper = np.empty((ny, nx), dtype=kb.dtype)
        taper[:ny // 2, :nx // 2] = quarter[::-1, ::-1]
        taper[ny // 2:, :nx // 2] = quarter[:, ::-1]
        taper[:ny // 2, nx // 2:] = quarter[::-1, :]
        taper[ny // 2:, nx // 2:] = quarter[:, :]

        # Normalise by oversampling factor
        taper *= cf.oversample**2

        return taper
    else:
        raise ValueError("Invalid filter_type '%s'" % filter_type)
