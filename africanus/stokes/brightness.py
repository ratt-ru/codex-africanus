#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def brightness(stokes, polarisation_type=None):
    """
    Computes the brightness matrix from the ``stokes`` parameters
    of a number of sources for the given ``polarisation_type``.

    ``stokes`` is an array of polarisations of shape :code:`(source, pol)`.
    The ``pol`` dimension must have size 1, 2 or 4.

    * If pol has size 1, this always represents the ``I`` polarisation
    * If pol has size 2, this represents :code:`(I,Q)` in the ``linear``
      case, or :code:`(I,V)` in the ``circular`` case.
    * If pol has size 4, this always represents :code:`(I,Q,U,V)`.

    Parameters
    ----------
    stokes : :class:`numpy.ndarray`
        floating point array of shape :code:`(source, pol)`
    polarisation_type : {'linear', 'circular'}
        Type of polarisation

    Returns
    -------
    :class:`numpy.ndarray`
        complex brightness matrix of shape :code:`(source, pol)`
    """
    nsource, npol = stokes.shape
    dtype = np.complex64 if stokes.dtype == np.float32 else np.complex128

    if polarisation_type is None:
        polarisation_type = 'linear'

    if npol == 1:
        return stokes.astype(dtype)
    elif polarisation_type == 'linear':
        if npol == 2:
            brightness = np.empty_like(stokes, dtype=dtype)
            brightness[:, 0] = stokes[:, 0] + stokes[:, 1]     # I + Q
            brightness[:, 1] = stokes[:, 0] - stokes[:, 1]     # I - Q
        elif npol == 4:
            brightness = np.empty_like(stokes, dtype=dtype)
            brightness[:, 0] = stokes[:, 0] + stokes[:, 1]     # I + Q
            brightness[:, 1] = stokes[:, 2] + 1j*stokes[:, 3]  # U + Vi
            brightness[:, 2] = stokes[:, 2] - 1j*stokes[:, 3]  # U - Vi
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
            brightness[:, 1] = stokes[:, 1] + 1j*stokes[:, 2]  # Q + Ui
            brightness[:, 2] = stokes[:, 1] - 1j*stokes[:, 2]  # Q - Ui
            brightness[:, 3] = stokes[:, 0] - stokes[:, 3]     # I - V
        else:
            raise ValueError("npol not in (1, 2, 4)")
    else:
        raise ValueError("polarisation_type must be 'linear' or 'circular'")

    return brightness
