# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np


def estimate_kaiser_bessel_beta(full_support):
    r"""
    Estimate the kaiser bessel beta using he following heuristic,
    with :math:`W` denoting ``full_support``:

    .. math::

        \beta = 1.2 \pi \sqrt{0.25 \text{ W }^2 - 1.0 }

    Parameters
    ----------
    full_support : int
        Full support of the filter

    Returns
    -------
    float
        kaiser Bessel beta shape parameter
    """

    # NOTE(bmerry)
    # Puts the first null of the taper function
    # at the edge of the image
    beta = np.pi * np.sqrt(0.25 * full_support**2 - 1.0)
    # Move the null outside the image,
    # to avoid numerical instabilities.
    # This will cause a small amount of aliasing at the edges,
    # which ideally should be handled by clipping the image.
    return beta * 1.2


def kaiser_bessel(taps, full_support, beta):
    r"""
    Compute a 1D Kaiser Bessel filter.

    Parameters
    ----------
    taps : :class:`numpy.ndarray`
        Filter position
    full_support : int
        Full support of the filter
    beta : float, optional
        Kaiser Bessel shape parameter

    Returns
    -------
    :class:`numpy.ndarray`
        Kaiser Bessel filter
    """

    # Sanity check
    M = full_support
    hM = M // 2
    assert np.all(-hM <= taps) & np.all(taps <= hM)

    param = 1 - (2 * taps / M)**2
    param[param < 0] = 0  # Zero negative numbers
    return np.i0(beta * np.sqrt(param)) / np.i0(beta)


def kaiser_bessel_fourier(n, full_support, beta):
    alpha = beta / np.pi
    inner = np.lib.scimath.sqrt((full_support * n)**2 - alpha * alpha)
    return full_support * np.sinc(inner).real / np.i0(beta)
