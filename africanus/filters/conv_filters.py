# -*- coding: utf-8 -*-


import collections

import numpy as np

from .kaiser_bessel_filter import (kaiser_bessel_with_sinc,
                                   estimate_kaiser_bessel_beta)

ConvolutionFilter = collections.namedtuple("ConvolutionFilter",
                                           ['half_sup', 'oversample',
                                            'full_sup_wo_padding', 'full_sup',
                                            'no_taps', 'filter_taps'])
"""
:class:`collections.namedtuple` containing attributes
defining a 2D Convolution Filter. A namedtuple is used
because they're easier to use when using
``nopython`` mode in :mod:`numba`.

.. attribute:: full_sup

    Full support

.. attribute:: full_sup_wo_padding

    Full support without padding

.. attribute:: half_sup

    Half support

.. attribute:: oversample

    Oversampling factor

.. attribute:: no_taps

    Number of taps

.. attribute:: filter_taps

    2D filter taps with shape (v, u)
"""


class AsymmetricKernel(Exception):
    pass


def convolution_filter(half_support, oversampling_factor,
                       filter_type, **kwargs):
    r"""
    Create a 2D Convolution Filter suitable
    for use with gridding and degridding functions.

    Parameters
    ----------
    half_support : integer
        Half support (N) of the filter. The filter has a
        full support of N*2 + 3 taps.
        Two of the taps exist as padding.
    oversampling_factor : integer
        Number of spaces in-between grid-steps
        (improves gridding/degridding accuracy)
    filter_type : {'kaiser-bessel', 'sinc'}
        Filter type. See `Convolution Filters <convolution-filter-api_>`_
        for further information.
    beta : float, optional
        Beta shape parameter for
        `Kaiser Bessel <kaiser-bessel-filter_>`_ filters.
    normalise : {True, False}
        Normalise the filter by the it's volume.
        Defaults to ``True``.

    Returns
    -------
    :class:`ConvolutionFilter`
        namedtuple containing filter attributes
    """
    full_sup_wo_padding = (half_support * 2 + 1)
    full_sup = full_sup_wo_padding + 2  # + padding
    no_taps = full_sup + (full_sup - 1) * (oversampling_factor - 1)

    normalise = kwargs.pop("normalise", True)

    taps = np.arange(no_taps) / oversampling_factor - full_sup // 2

    if filter_type == 'sinc':
        filter_taps = np.sinc(taps)
    elif filter_type == 'kaiser-bessel':
        # https://www.dsprelated.com/freebooks/sasp/Kaiser_Window.html
        try:
            beta = kwargs.pop('beta')
        except KeyError:
            beta = estimate_kaiser_bessel_beta(full_sup)

        # Compute Kaiser Bessel and multiply in the sinc
        filter_taps = kaiser_bessel_with_sinc(taps, full_sup,
                                              oversampling_factor, beta,
                                              normalise=normalise)
    else:
        raise ValueError("Expected one of {'kaiser-bessel', 'sinc'}")

    # Expand filter taps to 2D
    filter_taps = np.outer(filter_taps, filter_taps)

    if not np.all(filter_taps == filter_taps.T):
        raise AsymmetricKernel("Kernel is asymmetric")

    return ConvolutionFilter(half_support, oversampling_factor,
                             full_sup_wo_padding, full_sup,
                             no_taps, filter_taps)
