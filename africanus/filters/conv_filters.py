# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np

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


def convolution_filter(half_support, oversampling_factor,
                       filter_type, **kwargs):
    r"""
    Create a 2D Convolution Filter suitable
    for use with gridding and degridding functions.

    Parameters
    ----------
    half_support : integer
        Half support (N) of the filter. The filter has a
        full support of N*2 + 1 taps
    oversampling_factor : integer
        Number of spaces in-between grid-steps
        (improves gridding/degridding accuracy)
    filter_type : {'kaiser-bessel'}
        Filter type. See `Convolution Filters <convolution-filter-api_>`_
        for further information.
    beta : float, optional
        Beta shape parameter for
        `Kaiser Bessel <kaiser-bessel-filter_>`_ filters.
        If not provided, the following heuristic is used:

        .. math::

            beta = 1.2 \pi \sqrt{0.25 W^2 - 1.0 }

    Returns
    -------
    :class:`ConvolutionFilter`
        namedtuple containing filter attributes
    """
    full_sup_wo_padding = (half_support * 2 + 1)
    full_sup = full_sup_wo_padding + 2  # + padding
    no_taps = full_sup + (full_sup - 1) * (oversampling_factor - 1)

    taps = np.arange(no_taps) / oversampling_factor - full_sup // 2

    if filter_type == 'kaiser-bessel':
        # https://www.dsprelated.com/freebooks/sasp/Kaiser_Window.html
        try:
            beta = kwargs.pop('beta')
        except KeyError:
            # NOTE(bmerry)
            # Puts the first null of the taper function
            # at the edge of the image
            beta = np.pi * np.sqrt(0.25 * full_sup**2 - 1.0)
            # Move the null outside the image,
            # to avoid numerical instabilities.
            # This will cause a small amount of aliasing at the edges,
            # which ideally should be handled by clipping the image.
            beta *= 1.2

        # Sanity check
        M = full_sup
        hM = M // 2
        assert np.all(-hM <= taps) & np.all(taps <= hM)

        param = 1 - (2 * taps / M)**2
        param[param < 0] = 0  # Zero negative numbers
        filter_taps = np.i0(beta * np.sqrt(param)) / np.i0(beta)
        filter_taps /= np.trapz(filter_taps, taps)
    else:
        raise ValueError("Expected one of {'kaiser-bessel'}")

    # Expand filter taps to 2D
    filter_taps = np.outer(filter_taps, filter_taps)

    return ConvolutionFilter(half_support, oversampling_factor,
                             full_sup_wo_padding, full_sup,
                             no_taps, filter_taps)
