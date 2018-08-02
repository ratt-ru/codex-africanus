from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numba
import numpy as np


@numba.jit(nopython=True, nogil=True, cache=True)
def _nb_radec_to_lmn(radec, phase_centre):
    assert radec.ndim == 2 and radec.shape[1] == 2
    assert phase_centre.ndim == 1 and phase_centre.shape[0] == 2

    lmn = np.empty(shape=(radec.shape[0], 3), dtype=radec.dtype)

    pc_ra, pc_dec = phase_centre
    sin_d0 = np.sin(pc_dec)
    cos_d0 = np.cos(pc_dec)

    for s in range(radec.shape[0]):
        da = radec[s, 0] - pc_ra
        sin_da = np.sin(da)
        cos_da = np.cos(da)

        sin_d = np.sin(radec[s, 1])
        cos_d = np.cos(radec[s, 1])

        lmn[s, 0] = l = cos_d*sin_da
        lmn[s, 1] = m = sin_d*cos_d0 - cos_d*sin_d0*cos_da
        lmn[s, 2] = np.sqrt(1.0 - l**2 - m**2)

    return lmn


@numba.jit(nopython=True, nogil=True, cache=True)
def _nb_lm_to_radec(lmn, phase_centre):
    assert lmn.ndim == 2 and lmn.shape[1] == 3
    assert phase_centre.ndim == 1 and phase_centre.shape[0] == 2

    radec = np.empty(shape=(lmn.shape[0], 2), dtype=lmn.dtype)

    pc_ra, pc_dec = phase_centre
    sin_d0 = np.sin(pc_dec)
    cos_d0 = np.cos(pc_dec)

    for s in range(radec.shape[0]):
        l, m, n = lmn[s]

        radec[s, 1] = np.arcsin(m*cos_d0 + n*sin_d0)
        radec[s, 0] = pc_ra + np.arctan(l / (n*cos_d0 - m*sin_d0))

    return radec


def radec_to_lmn(radec, phase_centre=None):
    r"""
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
    radec : :class:`np.ndarray`
        radec coordinates of shape :code:`(coord, 2)`
        where Right-Ascension and Declination are in the
        last 2 components, respectively.
    phase_centre : :class:`np.ndarray`, optional
        radec coordinates of the Phase Centre.
        Shape :code:`(2,)`

    Returns
    -------
    :class:`np.ndarray`
        lm Direction Cosines of shape :code:`(coord, 2)`
    """
    if phase_centre is None:
        phase_centre = np.zeros((2,), dtype=radec.dtype)

    return _nb_radec_to_lmn(radec, phase_centre)


def lmn_to_radec(lmn, phase_centre=None):
    r"""
    Convert Direction Cosine lm coordinates to Right Ascension/Declination
    coordinates in radians, relative to the Phase Centre.

    .. math::
        :nowrap:

        \begin{eqnarray}
        & \delta = \, \arcsin \( m \cos \delta 0 + n \sin \delta 0 \)
        & \alpha = \, \arctan \( \frac{l, n \cos \delta 0 - m \sin \delta 0} \)
        \end{eqnarray}

    where :math:`\alpha` is the Right Ascension of each coordinate
    and the phase centre and :math:`\delta 0`
    is the Declination of the phase centre.

    Parameters
    ----------
    lmn : :class:`np.ndarray`
        lm Direction Cosines of shape :code:`(coord, 3)`
    phase_centre : :class:`np.ndarray`, optional
        radec coordinates of the Phase Centre.
        Shape :code:`(2,)`

    Returns
    -------
    :class:`np.ndarray`
        radec coordinates of shape :code:`(coord, 2)`
        where Right-Ascension and Declination are in the
        last 2 components, respectively.

    """
    if phase_centre is None:
        phase_centre = np.zeros((2,), dtype=lmn.dtype)

    return _nb_lm_to_radec(lmn, phase_centre)
