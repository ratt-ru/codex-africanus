# -*- coding: utf-8 -*-


import numpy as np

from africanus.util.docs import DocstringTemplate
from africanus.util.numba import generated_jit


@generated_jit(nopython=True, nogil=True, cache=True)
def feed_rotation(rotation_angles, rotation_angles_2=None, feed_type='linear'):
    pa_np_dtype = np.dtype(rotation_angles.dtype.name)
    dtype = np.result_type(pa_np_dtype, np.complex64)

    def impl(rotation_angles, rotation_angles_2=None, feed_type='linear'):
        if rotation_angles_2 is not None:
            if rotation_angles.shape != rotation_angles_2.shape:
                raise ValueError("parallactic angle shapes don't match")

            parangles2 = rotation_angles_2.ravel()

        parangles = rotation_angles.ravel()
        # Can't prepend shape tuple till the following is fixed
        # https://github.com/numba/numba/issues/5439
        # We know parangles.ndim == 1 though
        result = np.zeros((parangles.shape[0], 2, 2), dtype=dtype)

        # Linear feeds
        if feed_type == 'linear':
            for i in range(parangles.shape[0]):
                pa = parangles[i]
                pa_cos = np.cos(pa)
                pa_sin = np.sin(pa)

                if rotation_angles_2 is not None:
                    pa2 = parangles2[i]
                    pa2_cos = np.cos(pa2)
                    pa2_sin = np.sin(pa2)
                else:
                    pa2_cos = pa_cos
                    pa2_sin = pa_sin

                result[i, 0, 0] = pa_cos + 0j
                result[i, 0, 1] = pa_sin + 0j
                result[i, 1, 0] = -pa2_sin + 0j
                result[i, 1, 1] = pa2_cos + 0j

        # Circular feeds
        elif feed_type == 'circular':
            for i in range(parangles.shape[0]):
                pa = parangles[i]
                pa_cos = np.cos(pa)
                pa_sin = np.sin(pa)

                if rotation_angles_2 is not None:
                    pa2 = parangles2[i]
                    pa2_cos = np.cos(pa2)
                    pa2_sin = np.sin(pa2)
                else:
                    pa2_cos = pa_cos
                    pa2_sin = pa_sin

                result[i, 0, 0] = pa_cos - pa_sin*1j
                result[i, 0, 1] = 0.0
                result[i, 1, 0] = 0.0
                result[i, 1, 1] = pa2_cos + pa2_sin*1j
        else:
            raise ValueError("feed_type not in ('linear', 'circular')")

        return result.reshape(rotation_angles.shape + (2, 2))

    return impl


FEED_ROTATION_DOCS = DocstringTemplate(r"""
Computes the 2x2 feed rotation (L) matrix
from ``rotation_angles`` and ``rotation_angles_2``.

.. math::

    \textrm{linear}
    \begin{bmatrix}
    cos(ra) & sin(ra) \\
    -sin(ra2) & cos(ra2)
    \end{bmatrix}
    \qquad
    \textrm{circular}
    \begin{bmatrix}
    e^{-i ra} & 0 \\
    0 & e^{i ra2}
    \end{bmatrix}

Parameters
----------
rotation_angles : $(array_type)
    floating point rotation angles for the first receptor.
    Of shape :code:`(ra0, ra1, ..., ran)`.
rotation_angles_2 : $(array_type), optional
    floating point rotation angles for the second receptor.
    Of shape :code:`(ra0, ra1, ..., ran)`.
    If None, ``rotation_angles`` is substituted for this input.
feed_type : {'linear', 'circular'}
    The type of feed

Returns
-------
feed_matrix : $(array_type)
    Feed rotation matrix of shape :code:`(ra0, ra1,...,ran,2,2)`
""")

try:
    feed_rotation.__doc__ = FEED_ROTATION_DOCS.substitute(
                                array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass
