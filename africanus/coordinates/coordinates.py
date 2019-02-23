# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import wraps

import numpy as np
import numba

from africanus.util.docs import on_rtd, DocstringTemplate
from africanus.util.numba import is_numba_type_none
from africanus.util.requirements import requires_optional

try:
    from astropy.coordinates import CartesianRepresentation
except ImportError as e:
    opt_import_error = e
else:
    opt_import_error = None


@numba.jit(nopython=True, nogil=True, cache=True)
def _create_phase_centre(phase_centre, dtype):
    return np.zeros((2,), dtype=dtype)


@numba.jit(nopython=True, nogil=True, cache=True)
def _return_phase_centre(phase_centre, dtype):
    return phase_centre


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

            lmn[s, 0] = l = cos_d*sin_da  # noqa
            lmn[s, 1] = m = sin_d*cos_d0 - cos_d*sin_d0*cos_da
            lmn[s, 2] = np.sqrt(1.0 - l**2 - m**2)

        return lmn

    return _radec_to_lmn_impl


def radec_to_lm(radec, phase_centre=None):
    dtype = radec.dtype

    if is_numba_type_none(phase_centre):
        _maybe_create_phase_centre = _create_phase_centre
    else:
        _maybe_create_phase_centre = _return_phase_centre

    @wraps(radec_to_lm)
    def _radec_to_lm_impl(radec, phase_centre=None):
        sources, components = radec.shape

        if radec.ndim != 2 or components != 2:
            raise ValueError("radec must have shape (source, 2)")

        lm = np.empty(shape=(sources, 2), dtype=dtype)

        pc_ra, pc_dec = _maybe_create_phase_centre(phase_centre, dtype)
        sin_d0 = np.sin(pc_dec)
        cos_d0 = np.cos(pc_dec)

        for s in range(sources):
            da = radec[s, 0] - pc_ra
            sin_da = np.sin(da)
            cos_da = np.cos(da)

            sin_d = np.sin(radec[s, 1])
            cos_d = np.cos(radec[s, 1])

            lm[s, 0] = cos_d*sin_da
            lm[s, 1] = sin_d*cos_d0 - cos_d*sin_d0*cos_da

        return lm

    return _radec_to_lm_impl


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


def lm_to_radec(lm, phase_centre=None):
    dtype = lm.dtype

    if is_numba_type_none(phase_centre):
        _maybe_create_phase_centre = _create_phase_centre
    else:
        _maybe_create_phase_centre = _return_phase_centre

    @wraps(lm_to_radec)
    def _lm_to_radec_impl(lm, phase_centre=None):
        if lm.ndim != 2 or lm.shape[1] != 2:
            raise ValueError("lm must have shape (source, 2)")

        radec = np.empty(shape=(lm.shape[0], 2), dtype=dtype)

        pc_ra, pc_dec = _maybe_create_phase_centre(phase_centre, dtype)
        sin_d0 = np.sin(pc_dec)
        cos_d0 = np.cos(pc_dec)

        for s in range(radec.shape[0]):
            l, m = lm[s]
            n = np.sqrt(1.0 - l**2 - m**2)

            radec[s, 1] = np.arcsin(m*cos_d0 + n*sin_d0)
            radec[s, 0] = pc_ra + np.arctan(l / (n*cos_d0 - m*sin_d0))

        return radec

    return _lm_to_radec_impl


# inspect.getargspec doesn't work on a numba dispatcher object
# so rtd fails.
if not on_rtd():
    jitter = numba.generated_jit(nopython=True, nogil=True, cache=True)
    lmn_to_radec = jitter(lmn_to_radec)
    lm_to_radec = jitter(lm_to_radec)
    radec_to_lmn = jitter(radec_to_lmn)
    radec_to_lm = jitter(radec_to_lm)


@requires_optional("astropy", opt_import_error)
def astropy_radec_to_lmn(radec, phase_centre):
    """
    Astropy radec_to_lmn conversion, useful for testing.

    Parameters
    ----------
    radec : :class:`astropy.coordinates.SkyCoord`
        Sky coordinates
    phase_centre : :class:`astropy.coordinates.SkyCoord`
        Phase Centre

    Returns
    -------
    lmn : :class:`numpy.ndarray`
        lmn coordinates of shape :code:`(source, 3)`

    """
    # Transform radec relative to phase centre
    relative = radec.transform_to(phase_centre.skyoffset_frame())
    ret = relative.represent_as(CartesianRepresentation)

    # Rearrange astropy's coordinates into lmn convention
    result = np.empty((ret.x.value.shape[0], 3), dtype=ret.x.value.dtype)
    result[:, 0] = ret.y.value
    result[:, 1] = ret.z.value
    result[:, 2] = ret.x.value
    return result


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
    lm Direction Cosines of shape :code:`(coord, $(lm_components))`
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
$(lm_name) : $(array_type)
    lm Direction Cosines of shape :code:`(coord, $(lm_components))`
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
                                lm_components="3",
                                array_type=":class:`numpy.ndarray`")
    radec_to_lm.__doc__ = RADEC_TO_LMN_DOCS.substitute(
                                lm_components="2",
                                array_type=":class:`numpy.ndarray`")
    lmn_to_radec.__doc__ = LMN_TO_RADEC_DOCS.substitute(
                                lm_name="lmn", lm_components="3",
                                array_type=":class:`numpy.ndarray`")
    lm_to_radec.__doc__ = LMN_TO_RADEC_DOCS.substitute(
                                lm_name="lm", lm_components="2",
                                array_type=":class:`numpy.ndarray`")

except AttributeError:
    pass
