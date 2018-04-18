# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from operator import mul
try:
    from functools import reduce
except ImportError:
    pass

import numba
import numpy as np

from africanus.util import corr_shape

@numba.njit(nogil=True, cache=True)
def _nb_feed_rotation(parallactic_angles, feed_type, feed_rotation):
    shape = parallactic_angles.shape
    parangles = parallactic_angles.flat

    # Linear feeds
    if feed_type == 0:
        for i, pa in enumerate(parangles):
            pa_cos = np.cos(pa)
            pa_sin = np.sin(pa)

            feed_rotation.real[i,0,0] = pa_cos
            feed_rotation.imag[i,0,0] = 0.0
            feed_rotation.real[i,0,1] = pa_sin
            feed_rotation.imag[i,0,1] = 0.0
            feed_rotation.real[i,1,0] = -pa_sin
            feed_rotation.imag[i,1,0] = 0.0
            feed_rotation.real[i,1,1] = pa_cos
            feed_rotation.imag[i,1,1] = 0.0

    # Circular feeds
    elif feed_type == 1:
        for i, pa in enumerate(parangles):
            pa_cos = np.cos(pa)
            pa_sin = np.sin(pa)

            feed_rotation.real[i,0,0] = pa_cos
            feed_rotation.imag[i,0,0] = -pa_sin
            feed_rotation[i,0,1] = 0.0 + 0.0*1j
            feed_rotation[i,1,0] = 0.0 + 0.0*1j
            feed_rotation.real[i,1,1] = pa_cos
            feed_rotation.imag[i,1,1] = pa_sin
    else:
        raise ValueError("Invalid feed_type")

    return feed_rotation.reshape(shape + (2,2))

def feed_rotation(parallactic_angles, feed_type='linear'):
    """
    Computes the 2x2 feed rotation (L) matrix
    from the ``parallactic_angles``.

    Parameters
    ----------
    parallactic_angles : :class:`numpy.ndarray`
        floating point parallactic angles. Of shape
        :code:`(pa0, pa1, ..., pan)`.
    feed_type : {'linear', 'circular'}
        The type of feed

    Returns
    -------
    :class:`numpy.ndarray`
        Feed rotation matrix of shape :code:`(pa0, pa1,...,pan,2,2)
    """
    if feed_type == 'linear':
        poltype = 0
    elif feed_type == 'circular':
        poltype = 1
    else:
        raise ValueError("Invalid feed_type '%s'" % feed_type)

    if parallactic_angles.dtype == np.float32:
        dtype = np.complex64
    elif parallactic_angles.dtype == np.float64:
        dtype = np.complex128
    else:
        raise ValueError("parallactic_angles has "
                        "none-floating point type %s"
                            % parallactic_angles.dtype)

    # Create result array with flattened parangles
    shape = (reduce(mul, parallactic_angles.shape),) + (2,2)
    result = np.empty(shape, dtype=dtype)

    return _nb_feed_rotation(parallactic_angles, poltype, result)
