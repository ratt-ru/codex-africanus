from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np


def cell_size(nl, nm, umax, vmax):
    r"""
    Returns the cell size in ``l`` and ``m``,
    given the maximum ``U`` and ``V`` coordinate.


    Parameters
    ----------
    nl : int
        Number of pixels in ``l``
    nm : int
        Number of pixels in ``m``
    umax : float
        Maximum U coordinate
    vmax : float
        Maximum V coordinate


    Returns
    -------
    tuple
        Cell size in the l and m dimensions
    """
    return np.array([umax / (2.*nl), vmax / (2.*nm)])
