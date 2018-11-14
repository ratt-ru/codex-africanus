# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import wraps

import numba
import numpy as np

from ..util.docs import DocstringTemplate
from ..util.numba import is_numba_type_none


@numba.jit(nopython=True, nogil=True, cache=True)
def _create_phase_centre(phase_centre, dtype):
    return np.zeros((2,), dtype=dtype)


@numba.jit(nopython=True, nogil=True, cache=True)
def _return_phase_centre(phase_centre, dtype):
    return phase_centre


@numba.generated_jit(nopython=True, nogil=True, cache=True)
def radec_to_lmn(radec, phase_centre=None):
    dtype = radec.dtype

    if is_numba_type_none(phase_centre):
        _maybe_create_phase_centre = _create_phase_centre
    else:
        _maybe_create_phase_centre = _return_phase_centre

    @wraps(radec_to_lmn)
    def _radec_to_lmn_impl(radec, phase_centre=None):
        sources, components = radec.shape

        if radec.ndim != 2 or components != 2:
            raise ValueError("radec must have shape (source, 2)")

        lmn = np.empty(shape=(sources, 3), dtype=dtype)

        pc_ra, pc_dec = _maybe_create_phase_centre(phase_centre, dtype)
        sin_d0 = np.sin(pc_dec)
        cos_d0 = np.cos(pc_dec)

        for s in range(sources):
            da = radec[s, 0] - pc_ra
            sin_da = np.sin(da)
            cos_da = np.cos(da)

            sin_d = np.sin(radec[s, 1])
            cos_d = np.cos(radec[s, 1])

            lmn[s, 0] = l = cos_d*sin_da
            lmn[s, 1] = m = sin_d*cos_d0 - cos_d*sin_d0*cos_da
            lmn[s, 2] = np.sqrt(1.0 - l**2 - m**2)

        return lmn

    return _radec_to_lmn_impl


@numba.generated_jit(nopython=True, nogil=True, cache=True)
def lmn_to_radec(lmn, phase_centre=None):
    dtype = lmn.dtype

    if is_numba_type_none(phase_centre):
        _maybe_create_phase_centre = _create_phase_centre
    else:
        _maybe_create_phase_centre = _return_phase_centre

    @wraps(lmn_to_radec)
    def _lmn_to_radec_impl(lmn, phase_centre=None):
        if lmn.ndim != 2 or lmn.shape[1] != 3:
            raise ValueError("lmn must have shape (source, 3)")

        radec = np.empty(shape=(lmn.shape[0], 2), dtype=dtype)

        pc_ra, pc_dec = _maybe_create_phase_centre(phase_centre, dtype)
        sin_d0 = np.sin(pc_dec)
        cos_d0 = np.cos(pc_dec)

        for s in range(radec.shape[0]):
            l, m, n = lmn[s]

            radec[s, 1] = np.arcsin(m*cos_d0 + n*sin_d0)
            radec[s, 0] = pc_ra + np.arctan(l / (n*cos_d0 - m*sin_d0))

        return radec

    return _lmn_to_radec_impl


RADEC_TO_LMN_DOCS = DocstringTemplate(r"""
Converts Right-Ascension/Declination coordinates in radians
to a Direction Cosine lm coordinates, relative to the Phase Centre.

.. math::
    :nowrap:

    \begin{eqnarray}
        & l =& \, \cos \, \delta  \sin \, \Delta \alpha  \\
        & m =& \, \sin \, \delta \cos \, \delta 0 -
                     \cos \delta \sin \delta 0 \cos \Delta \alpha \\
        & n =& \, \sqrt{1 - l^2 - m^2} - 1
    \end{eqnarray}

where :math:`\Delta \alpha = \alpha - \alpha 0` is the difference between
the Right Ascension of each coordinate and the phase centre and
:math:`\delta 0` is the Declination of the phase centre.

Parameters
----------
radec : $(array_type)
    radec coordinates of shape :code:`(coord, 2)`
    where Right-Ascension and Declination are in the
    last 2 components, respectively.
phase_centre : $(array_type), optional
    radec coordinates of the Phase Centre.
    Shape :code:`(2,)`

Returns
-------
$(array_type)
    lm Direction Cosines of shape :code:`(coord, 2)`
""")


LMN_TO_RADEC_DOCS = DocstringTemplate(r"""
Convert Direction Cosine lm coordinates to Right Ascension/Declination
coordinates in radians, relative to the Phase Centre.

.. math::
    :nowrap:

    \begin{eqnarray}
    & \delta = & \, \arcsin \left( m \cos \delta 0 +
                                 n \sin \delta 0 \right) \\
    & \alpha = & \, \arctan \left( \frac{l}{n \cos \delta 0 -
                                 m \sin \delta 0} \right) \\
    \end{eqnarray}

where :math:`\alpha` is the Right Ascension of each coordinate
and the phase centre and :math:`\delta 0`
is the Declination of the phase centre.

Parameters
----------
lmn : $(array_type)
    lm Direction Cosines of shape :code:`(coord, 3)`
phase_centre : $(array_type), optional
    radec coordinates of the Phase Centre.
    Shape :code:`(2,)`

Returns
-------
$(array_type)
    radec coordinates of shape :code:`(coord, 2)`
    where Right-Ascension and Declination are in the
    last 2 components, respectively.

""")

try:
    radec_to_lmn.__doc__ = RADEC_TO_LMN_DOCS.substitute(
                                array_type=":class:`numpy.ndarray`")
    lmn_to_radec.__doc__ = LMN_TO_RADEC_DOCS.substitute(
                                array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass
