# -*- coding: utf-8 -*-


import numpy as np


def abs_diff(x, xp):
    """
    Gets matrix of differences between
    :math:`D`-dimensional vectors x and xp
    i.e.

    .. math::
        X_{ij} = |x_i - x_j|

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        Array of inputs of shape :code:`(N, D)`.
    xp : :class:`numpy.ndarray`
        Array of inputs of shape :code:`(Np, D)`.

    Returns
    -------
    XX : :class:`numpy.ndarray`
        Array of differences of shape :code:`(N, Np)`.

    """
    try:
        N, D = x.shape
        Np, D = xp.shape
    except Exception:
        N = x.size
        D = 1
        Np = xp.size
        x = np.reshape(x, (N, D))
        xp = np.reshape(xp, (Np, D))
    xD = np.zeros([D, N, Np])
    xpD = np.zeros([D, N, Np])
    for i in range(D):
        xD[i] = np.tile(x[:, i], (Np, 1)).T
        xpD[i] = np.tile(xp[:, i], (N, 1))
    return np.linalg.norm(xD - xpD, axis=0)
