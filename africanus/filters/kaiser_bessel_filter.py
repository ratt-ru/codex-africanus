# -*- coding: utf-8 -*-


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

    param = 1 - (2 * u / W)**2
    param[param < 0] = 0  # Zero negative numbers
    return np.i0(beta * np.sqrt(param)) / np.i0(beta)


def kaiser_bessel_with_sinc(u, W, oversample, beta, normalise=True):
    """
    Produces a filter composed of Kaiser Bessel multiplied by a sinc.

    Accounts for the oversampling factor, as well as normalising the filter.

    Parameters
    ----------
    u : :class:`numpy.ndarray`
        Filter positions
    W : int
        Width of the filter
    oversample : int
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
    kb = kaiser_bessel(u, W, beta)
    kb *= oversample
    kb *= np.sinc(u / oversample)

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
    term = (np.pi*W*x)**2 - beta**2
    val = np.lib.scimath.sqrt(term).real
    return np.sin(val)/val
