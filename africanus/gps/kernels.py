# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from africanus.gps.utils import abs_diff


def exponential_squared(x, xp, sigmaf, l):
    """
    Create exponential squared covariance
    function between :math:`D` dimensional
    vectors :math:`x` and :math:`x_p` i.e.

    .. math::
        k(x, x_p) = \\sigma_f^2 \\exp\\left(-\\frac{(x-x_p)^2}{2l^2}\\right)

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        Array of shape :code:`(N, D)`.
    xp : :class:`numpy.ndarray`
        Array of shape :code:`(Np, D)`.
    sigmaf : float
        The signal variance hyper-parameter
    l : float
        The length scale hyper-parameter
    
    Returns
    -------
    K : :class:`numpy.ndarray`
        Array of shape :code:`(N, Np)`
    """
    xxp = abs_diff(x, xp)
    return sigmaf**2*np.exp(-xxp**2/(2*l**2))
