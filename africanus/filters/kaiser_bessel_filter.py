# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np


def estimate_kaiser_bessel_beta(W):
    r"""
    Estimate the kaiser bessel beta using the following heuristic:

    .. math::

        \beta = 2.34 \times W

    Derived from `Nonuniform fast Fourier transforms
    using min-max interpolation
    <nufft-min-max-ref_>`_.

    .. _nufft-min-max-ref: https://ieeexplore.ieee.org/document/1166689/

    Parameters
    ----------
    W : int
        Width of the filter

    Returns
    -------
    float
        Kaiser Bessel beta shape parameter
    """
    return 2.34*W


def kaiser_bessel(u, W, beta):
    r"""
    Compute a 1D Kaiser Bessel filter as defined
    in `Selection of a Convolution Function
    for Fourier Inversion Using Gridding
    <kbref_>`_.

    .. _kbref: https://ieeexplore.ieee.org/document/97598/

    Parameters
    ----------
    u : :class:`numpy.ndarray`
        Filter positions
    W : int
        Width of the filter
    beta : float, optional
        Kaiser Bessel shape parameter

    Returns
    -------
    :class:`numpy.ndarray`
        Kaiser Bessel filter with the same shape as `u`
    """

    # Sanity check
    hW = W // 2
    assert np.all(-hW <= u) & np.all(u <= hW)

    param = 1.0 - (2.0 * u / W)**2

    if not np.all(param >= 0.0):
        raise ValueError("Illegal filter positions %s" % param)

    return np.i0(beta * np.sqrt(param)) / np.i0(beta)


def kaiser_bessel_with_sinc(full_support, oversampling, beta, normalise=False):
    """
    Produces a filter composed of Kaiser Bessel multiplied by a sinc.

    Accounts for the oversampling factor, as well as normalising the filter.

    Parameters
    ----------
    full_support : int
        Full support of the filter
    oversampling : int
        Oversampling factor
    beta : float
        Kaiser Bessel shape parameter
    normalise : optional, {True, False}
        True if the filter should be normalised

    Returns
    -------
    :class:`numpy.ndarray`
        Filter with the same shape as `u`
    """
    W = full_support * oversampling
    u = np.arange(W, dtype=np.float64) - W // 2

    sinc = np.sinc(u / oversampling)
    kb = sinc * kaiser_bessel(u, W, beta)

    if normalise:
        kb /= np.trapz(kb, u)

    return kb


def kaiser_bessel_fourier(x, W, beta):
    r"""
    Computes the Fourier Transform of a 1D Kaiser Bessel filter.
    as defined in `Selection of a Convolution Function
    for Fourier Inversion Using Gridding
    <kbref_>`_.

    .. _kbref: https://ieeexplore.ieee.org/document/97598/


    Parameters
    ----------
    x : :class:`numpy.ndarray`
        Filter positions
    W : int
        Width of the filter.
    beta : float
        Kaiser bessel shape parameter

    Returns
    -------
    :class:`numpy.ndarray`
        Fourier Transform of the Kaiser Bessel,
        with the same shape as `x`.
    """
    from scipy.special import jv as bessel1
    term = (np.pi*W*x)**2 - beta**2
    val = np.lib.scimath.sqrt(term)
    Lambda = np.sqrt(2.0/val)*bessel1(0.5, val)
    return np.sqrt(np.pi)*W*Lambda/(2.0*np.i0(beta))


def wsclean_kaiser_bessel_with_sinc(filter_support, oversample, beta):
    """
    Reproduction of wsclean's WStackingGridder::makeKaiserBesselKernel
    """
    W = filter_support*oversample
    hW = W // 2
    filter_ratio = 1.0 / np.float64(oversample)

    half_sinc = np.empty(hW+1, dtype=np.float64)
    x = np.arange(0, hW+1, dtype=np.float64)

    # Assumption: oversample is not placed in the divisor because
    # it is factored out by norm_factor
    half_sinc[0] = 1.0 / oversample

    for i in range(1, hW+1):
        x = float(i)
        half_sinc[i] = np.sin(np.pi*x/oversample) / (np.pi*x)

    norm_factor = np.float64(oversample) / np.i0(beta)
    kernel = np.empty(W, dtype=np.float64)

    for i in range(0, hW+1):
        term = float(i) / hW
        term = beta * np.sqrt(1.0 - (term*term))
        kernel[hW + i] = half_sinc[i] * np.i0(term) * norm_factor

    # term = x / hW
    # term *= term
    # term[:] = -term
    # term += 1.0

    # kernel[hW:] = half_sinc * np.i0(beta * np.sqrt(term)) * norm_factor

    for i in range(0, hW+1):
        kernel[i] = kernel[W - 1 - i]

    return kernel
