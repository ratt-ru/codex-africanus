# -*- coding: utf-8 -*-


from functools import reduce
from operator import mul

import numpy as np

from africanus.config import config
from africanus.util.docs import DocstringTemplate
from africanus.util.numba import generated_jit


@generated_jit(nopython=True, nogil=True, cache=True)
def feed_rotation(parallactic_angles, feed_type='linear'):
    pa_np_dtype = np.dtype(parallactic_angles.dtype.name)
    dtype = np.result_type(pa_np_dtype, np.complex64)

    def impl(parallactic_angles, feed_type='linear'):
        parangles = parallactic_angles.flat

        shape = (reduce(lambda x, y: x*y, parallactic_angles.shape, 1),)
        result = np.zeros(shape + (2, 2), dtype=dtype)

        # Linear feeds
        if feed_type == 'linear':
            for i in range(shape[0]):
                pa = parangles[i]
                pa_cos = np.cos(pa)
                pa_sin = np.sin(pa)

                result[i, 0, 0] = pa_cos + 0j
                result[i, 0, 1] = pa_sin + 0j
                result[i, 1, 0] = -pa_sin + 0j
                result[i, 1, 1] = pa_cos + 0j

        # Circular feeds
        elif feed_type == 'circular':
            for i in range(shape[0]):
                pa = parangles[i]
                pa_cos = np.cos(pa)
                pa_sin = np.sin(pa)

                result[i, 0, 0] = pa_cos - pa_sin*1j
                result[i, 0, 1] = 0.0
                result[i, 1, 0] = 0.0
                result[i, 1, 1] = pa_cos + pa_sin*1j
        else:
            raise ValueError("feed_type not in ('linear', 'circular')")

        return result.reshape(parallactic_angles.shape + (2, 2))

    return impl


FEED_ROTATION_DOCS = DocstringTemplate(r"""
Computes the 2x2 feed rotation (L) matrix
from the ``parallactic_angles``.

.. math::

    \textrm{linear}
    \begin{bmatrix}
    cos(pa) & sin(pa) \\
    -sin(pa) & cos(pa)
    \end{bmatrix}
    \qquad
    \textrm{circular}
    \begin{bmatrix}
    e^{-i pa} & 0 \\
    0 & e^{i pa}
    \end{bmatrix}

Parameters
----------
parallactic_angles : $(array_type)
    floating point parallactic angles. Of shape
    :code:`(pa0, pa1, ..., pan)`.
feed_type : {'linear', 'circular'}
    The type of feed

Returns
-------
feed_matrix : $(array_type)
    Feed rotation matrix of shape :code:`(pa0, pa1,...,pan,2,2)`
""")

try:
    feed_rotation.__doc__ = FEED_ROTATION_DOCS.substitute(
                                array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass
