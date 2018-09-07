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

        \beta = 2.34 \times W

    Derived from `Nonuniform fast Fourier transforms
    using min-max interpolation
    <https://ieeexplore.ieee.org/document/1166689/>`_.

    Parameters
    ----------
    full_support : int
        Full support of the filter

    Returns
    -------
    float
        Kaiser Bessel beta shape parameter
    """

    return 2.34*full_support


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
        Kaiser Bessel filter of shape :code:`(taps,)`
    """

    # Sanity check
    M = full_support
    hM = M // 2
    assert np.all(-hM <= taps) & np.all(taps <= hM)

    param = 1 - (2 * taps / M)**2
    param[param < 0] = 0  # Zero negative numbers
    return np.i0(beta * np.sqrt(param)) / np.i0(beta)


def kaiser_bessel_fourier(taps, npix, beta):
    r"""
    Computes the Fourier Transform of a 1D Kaiser Bessel filter.

    Parameters
    ----------
    positions : :class:`numpy.ndarray`
        Filter positions
    npix : int
        Number of pixels.
    beta : float
        Kaiser bessel shape parameter

    Returns
    -------
    :class:`numpy.ndarray`
        Array of shape :code:`(taps,)
    """
    term = (np.pi*npix*taps)**2 - beta**2
    val = np.lib.scimath.sqrt(term).real
    val = np.sqrt(term)
    return np.sin(val)/val
