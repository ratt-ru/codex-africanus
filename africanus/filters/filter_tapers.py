# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .kaiser_bessel_filter import (kaiser_bessel_fourier,
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
    if filter_type == "sinc":
        raise NotImplementedError("Please implement sinc support")
    elif filter_type == "kaiser-bessel":
        try:
            beta = kwargs.pop('beta')
        except KeyError:
            beta = estimate_kaiser_bessel_beta(conv_filter.full_sup)

        py = np.arange(ny) / ny - 0.5
        y = kaiser_bessel_fourier(py, conv_filter.full_sup, beta)
        y *= np.sinc(y / conv_filter.oversample)

        px = np.arange(nx) / nx - 0.5
        x = kaiser_bessel_fourier(px, conv_filter.full_sup, beta)
        x *= np.sinc(x / conv_filter.oversample)

        # Produce 2D taper
        return y[:, None] * x[None, :]
    else:
        raise ValueError("Invalid filter_type '%s'" % filter_type)
