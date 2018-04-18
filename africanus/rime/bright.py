#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from ..util import corr_shape as corr_shape_fn


def brightness(stokes, polarisation_type=None, corr_shape=None):
    """
    Computes the brightness matrix (B) from the ``stokes`` parameters
    of a number of sources for the given ``polarisation_type``.

    ``stokes`` is an array of polarisations of shape
    :code:`(dim_1, dim_2, ..., dim_n, pol)`.
    The ``pol`` dimension must have size 1, 2 or 4.

    * If pol has size 1, this always represents the ``I`` polarisation
    * If pol has size 2, this represents :code:`(I,Q)` in the ``linear``
      case, or :code:`(I,V)` in the ``circular`` case.
    * If pol has size 4, this always represents :code:`(I,Q,U,V)`.

    Parameters
    ----------
    stokes : :class:`numpy.ndarray`
        floating point array of shape :code:`(dim_1, dim_2, ..., dim_n, pol)`
    polarisation_type : {'linear', 'circular'}
        Type of polarisation. Defaults to linear
    corr_shape : {'matrix', 'flat'}
        The shape of the resulting correlations in the last
        dimensions in the result.
        If ``matrix``, the last two dimensions will be a matrix.
        If ``flat``, the last dimension will equal ``pol``.

    Returns
    -------
    :class:`numpy.ndarray`
        complex brightness matrix of shape
        :code:`(dim_1, dim_2, ..., dim_n, corr_1, corr_2)`
    """
    dtype = np.complex64 if stokes.dtype == np.float32 else np.complex128

    head, tail = stokes.shape[:-1], stokes.shape[-1]
    stokes = stokes.reshape((-1, tail))
    nsource, npol = stokes.shape

    if polarisation_type is None:
        polarisation_type = 'linear'

    if corr_shape is None:
        corr_shape = 'matrix'

    if npol == 1:
        brightness = stokes.astype(dtype)
    elif polarisation_type == 'linear':
        if npol == 2:
            brightness = np.empty_like(stokes, dtype=dtype)
            brightness[:, 0] = stokes[:, 0] + stokes[:, 1]     # I + Q
            brightness[:, 1] = stokes[:, 0] - stokes[:, 1]     # I - Q
        elif npol == 4:
            brightness = np.empty_like(stokes, dtype=dtype)
            brightness[:, 0] = stokes[:, 0] + stokes[:, 1]     # I + Q
            brightness[:, 1] = stokes[:, 2] + 1j * stokes[:, 3]  # U + Vi
            brightness[:, 2] = stokes[:, 2] - 1j * stokes[:, 3]  # U - Vi
            brightness[:, 3] = stokes[:, 0] - stokes[:, 1]     # I - Q
        else:
            raise ValueError("npol not in (1, 2, 4)")
    elif polarisation_type == 'circular':
        if npol == 2:
            brightness = np.empty_like(stokes, dtype=dtype)
            brightness[:, 0] = stokes[:, 0] + stokes[:, 1]     # I + V
            brightness[:, 1] = stokes[:, 0] - stokes[:, 1]     # I - V
        elif npol == 4:
            brightness = np.empty_like(stokes, dtype=dtype)
            brightness[:, 0] = stokes[:, 0] + stokes[:, 3]     # I + V
            brightness[:, 1] = stokes[:, 1] + 1j * stokes[:, 2]  # Q + Ui
            brightness[:, 2] = stokes[:, 1] - 1j * stokes[:, 2]  # Q - Ui
            brightness[:, 3] = stokes[:, 0] - stokes[:, 3]     # I - V
        else:
            raise ValueError("npol not in (1, 2, 4)")
    else:
        raise ValueError("polarisation_type must be 'linear' or 'circular'")

    return brightness.reshape(head + corr_shape_fn(npol, corr_shape))
