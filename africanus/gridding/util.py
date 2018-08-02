from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np


def cell_size(nl, nm, umax, vmax):
    r"""
    Returns the cell size in ``l`` and ``m``,
    given the size of the grid
    and the maximum ``U`` and ``V`` coordinate.


    Parameters
    ----------
    nl : int
        Number of pixels in ``l``
    nm : int
        Number of pixels in ``m``
    umax : float
        Maximum ``U`` coordinate in wavelengths.
    vmax : float
        Maximum ``V`` coordinate in wavelengths.

    Returns
    -------
    tuple
        Cell size of the l and m dimensions
        of shape :code:`(2,)`
    """
    return np.array([umax / (2.*nl), vmax / (2.*nm)])


_ARCSEC2RAD = np.deg2rad(1.0/(60*60))


def uv_scale(nl, nm, umax, vmax):
    r"""
    Returns the UV scaling parameters given
    the size of the grid in ``l`` and ``m``
    and the maximum ``U`` and ``V`` coordinate.

    Used for converting UVW coordinates in wavelengths
    to grid space in order to satisfy the
    `Similarity Theorem
    <https://www.cv.nrao.edu/course/astr534/FTSimilarity.html>`_.


    Parameters
    ----------
    nl : int
        Number of pixels in ``l``
    nm : int
        Number of pixels in ``m``
    umax : float
        Maximum ``U`` coordinate in wavelengths.
    vmax : float
        Maximum ``V`` coordinate in wavelengths.

    Returns
    -------
    tuple
        UV scaling parameters of shape :code:`(2,)`
    """
    return _ARCSEC2RAD * cell_size(nl, nm, umax, vmax) * np.array([nl, nm])
