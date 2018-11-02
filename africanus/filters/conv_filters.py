# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np

from .kaiser_bessel_filter import (kaiser_bessel_with_sinc,
                                   estimate_kaiser_bessel_beta)

ConvolutionFilter = collections.namedtuple(
                        "ConvolutionFilter",
                        ['full_support', 'oversampling', 'filter'])
"""
:class:`collections.namedtuple` containing attributes
defining a 2D Convolution Filter. A namedtuple is used
because they're easier to use when using
``nopython`` mode in :mod:`numba`.

.. attribute:: full_support

    Full support

.. attribute:: oversampling

    Oversampling factor

.. attribute:: filter

    2D filter with shape (v, u)
"""


class AsymmetricKernel(Exception):
    pass


def convolution_filter(filter_type, full_support, oversampling, **kwargs):
    r"""
    Create a 2D Convolution Filter suitable
    for use with gridding and degridding functions.

    Parameters
    ----------
    full_support : integer
        Full support (N) of the filter.
    oversampling : integer
        Number of spaces in-between grid-steps
        (improves gridding/degridding accuracy)
    filter_type : {'kaiser-bessel', 'sinc'}
        Filter type. See `Convolution Filters <convolution-filter-api_>`_
        for further information.
    normalise : {True, False}
        Normalise the filter by the it's volume.
        Defaults to ``False``.
    beta : float, optional
        Beta shape parameter for
        `Kaiser Bessel <kaiser-bessel-filter_>`_ filters.

    Returns
    -------
    :class:`ConvolutionFilter`
        namedtuple containing filter attributes
    """

    if full_support % 2 == 0:
        raise ValueError("full_support (%d) must be odd" % full_support)

    normalise = kwargs.pop("normalise", False)

    if filter_type == 'sinc':
        W = full_support * oversampling
        u = np.arange(W, dtype=np.float64) - W // 2
        conv_filter = np.sinc(u / oversampling)
    elif filter_type == 'kaiser-bessel':
        # https://www.dsprelated.com/freebooks/sasp/Kaiser_Window.html
        try:
            beta = kwargs.pop('beta')
        except KeyError:
            beta = estimate_kaiser_bessel_beta(full_support)

        # Compute Kaiser Bessel and multiply in the sinc
        conv_filter = kaiser_bessel_with_sinc(full_support,
                                              oversampling, beta,
                                              normalise=normalise)
    else:
        raise ValueError("Expected one of {'kaiser-bessel', 'sinc'}")

    # Expand filter taps to 2D
    conv_filter = np.outer(conv_filter, conv_filter)

    if not np.all(conv_filter == conv_filter.T):
        raise AsymmetricKernel("Kernel is asymmetric")

    return ConvolutionFilter(full_support, oversampling, conv_filter)
