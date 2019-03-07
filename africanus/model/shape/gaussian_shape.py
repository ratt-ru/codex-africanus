# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numba

from africanus.util.docs import DocstringTemplate
from africanus.constants import c as lightspeed


@numba.generated_jit(nopython=True, nogil=True, cache=True)
def gaussian(uvw, frequency, shape_params):
    fwhmint = 1.0 / np.sqrt(np.log(256))
    gauss_scale = fwhmint*np.sqrt(2.0)*np.pi/lightspeed

    dtype = np.result_type(*(np.dtype(a.dtype.name) for
                             a in (uvw, frequency, shape_params)))

    def impl(uvw, frequency, shape_params):
        nsrc = shape_params.shape[0]
        nrow = uvw.shape[0]
        nchan = frequency.shape[0]

        shape = np.empty((nsrc, nrow, nchan), dtype=dtype)

        for s in range(shape_params.shape[0]):
            emaj, emin, orientation = shape_params[s]

            el = emaj * np.sin(orientation)
            em = emaj * np.cos(orientation)
            er = emaj / (1.0 if emin == 0.0 else emin)

            for r in range(uvw.shape[0]):
                u, v, w = uvw[r]

                u1 = (u*em - v*el)*er
                v1 = u*el + v*em

                for f in range(frequency.shape[0]):
                    scaled_freq = frequency[f]*gauss_scale
                    fu1 = u1*scaled_freq
                    fv1 = v1*scaled_freq

                    shape[s, r, f] = np.exp(-(fu1*fu1 + fv1*fv1))

        return shape

    return impl


GAUSSIAN_DOCS = DocstringTemplate("""
Computes the Gaussian Shape Function.

Parameters
----------
uvw : $(array_type)
    UVW coordinates of shape :code:`(row, 3)`
frequency : $(array_type)
    frequencies of shape :code:`(chan,)`
shape_param : $(array_type)
    Gaussian Shape Parameters of shape :code:`(source, 3)`
    where the second dimension contains the
    `(emajor, eminor, orientation)` parameters describing
    the shape of the Gaussian

Returns
-------
gauss_shape : $(array_type)
    Shape parameters of shape :code:`(source, row, chan)`
""")

try:
    gaussian.__doc__ = GAUSSIAN_DOCS.substitute(
                            array_type=":class:`numpy.ndarray`")
except KeyError:
    pass
