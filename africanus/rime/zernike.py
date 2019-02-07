# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numba
import numpy as np

from africanus.util.docs import DocstringTemplate


@numba.jit(nogil=True, nopython=True, cache=True)
def fac(x):
    if x < 0:
        raise ValueError("Factorial input is negative.")
    if x == 0:
        return 1
    factorial = 1
    for i in range(1, x + 1):
        factorial *= i
    return factorial


@numba.jit(nogil=True, nopython=True, cache=True)
def pre_fac(k, n, m):
    numerator = (-1.0)**k * fac(n-k)
    denominator = (fac(k) * fac((n+m)/2.0 - k) * fac((n-m)/2.0 - k))
    return numerator / denominator


@numba.jit(nogil=True, nopython=True, cache=True)
def zernike_rad(m, n, rho):
    if (n < 0 or m < 0 or abs(m) > n):
        raise ValueError("m and n values are incorrect.")
    radial_component = 0
    for k in range((n-m)/2+1):
        radial_component += pre_fac(k, n, m) * rho ** (n - 2.0 * k)
    return radial_component


@numba.jit(nogil=True, nopython=True, cache=True)
def zernike(j, rho, phi):
    if rho > 1:
        return 0 + 0j
    j += 1
    n = 0
    j1 = j-1
    while (j1 > n):
        n += 1
        j1 -= n
    m = (-1)**j * ((n % 2) + 2 * int((j1+((n+1) % 2)) / 2.0))
    if (m > 0):
        return zernike_rad(m, n, rho) * np.cos(m * phi)
    if (m < 0):
        return zernike_rad(-m, n, rho) * np.sin(-m * phi)
    return zernike_rad(0, n, rho)


@numba.jit(nogil=True, nopython=True, cache=True)
def nb_zernike_dde(coords, coeffs, noll_index, out):
    sources, times, ants, chans, corrs = out.shape
    npoly = coeffs.shape[2]

    for s in range(sources):
        for t in range(times):
            for a in range(ants):
                for c in range(chans):
                    l, m, _ = coords[:, s, t, a, c]
                    rho = np.sqrt(l**2 + m**2)
                    phi = np.arctan2(l, m)

                    for p in range(npoly):
                        for co in range(corrs):
                            zc = coeffs[a, c, p, co]
                            zn = noll_index[a, c, p, co]
                            out[s, t, a, c, co] += zc * zernike(zn, rho, phi)

    return out


def zernike_dde(coords, coeffs, noll_index):
    """ Wrapper for :func:`nb_zernike_dde` """
    _, sources, times, ants, chans = coords.shape
    # ant, chan, poly, corr_1, ..., corr_n
    corr_shape = coeffs.shape[3:]
    npoly = coeffs.shape[2]

    # Flatten correlation dimensions for numba function
    fcorrs = np.product(corr_shape)
    ddes = np.zeros((sources, times, ants, chans, fcorrs), coeffs.dtype)

    coeffs = coeffs.reshape((ants, chans, npoly, fcorrs))
    noll_index = noll_index.reshape((ants, chans, npoly, fcorrs))

    result = nb_zernike_dde(coords, coeffs, noll_index, ddes)

    # Reshape to full correlation size
    return result.reshape((sources, times, ants, chans) + corr_shape)


_ZERNIKE_DOCSTRING = DocstringTemplate(r"""
Computes Direction Dependent Effects by evaluating
`Zernike Polynomials <zernike_wiki_>`_
defined by coefficients ``coeffs``
and noll indexes ``noll_index``
at the specified coordinates ``coords``.

Decomposition of a voxel beam cube into Zernike
polynomial coefficients can be achieved through the
use of the eidos_ package.

.. _zernike_wiki: https://en.wikipedia.org/wiki/Zernike_polynomials
.. _eidos: https://github.com/kmbasad/eidos/

Parameters
---------------
coords : $(array_type)
   Float coordinates at which to evaluate the zernike polynomials.
   Has shape :code:`(3, source, time, ant, chan)`. The three components in
   the first dimension represent
   l, m and frequency coordinates, respectively.
coeffs : $(array_type)
  complex Zernike polynomial coefficients.
  Has shape :code:`(ant, chan, corr_1, ..., corr_n, poly)`
  where ``poly`` is the number of polynomial coefficients
  and ``corr_1, ..., corr_n`` are a variable number of
  correlation dimensions.
noll_index : $(array_type)
  Noll index associated with each polynomial coefficient.
  Has shape :code:`(ant, chan, corr_1, ..., corr_n, poly)`.

Returns
----------
ddes : $(array_type)
   complex values with shape
   :code:`(source, time, ant, chan, corr_1, ..., corr_n)`
""")

try:
    zernike_dde.__doc__ = _ZERNIKE_DOCSTRING.substitute(
                                array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass
