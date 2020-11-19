# -*- coding: utf-8 -*-


import numpy as np
from africanus.gps.utils import abs_diff


def exponential_squared(x, xp, sigmaf, l, pspec=False):  # noqa: E741
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
    if pspec:
        N, D = x.shape
        if D != 1:
            raise(NotImplementedError, "Only 1D pspecs supported")
        if (x != xp).any():
            raise(ValueError, "pspec only defined if x = xp")
        x = x.squeeze()
        delx = x[1] - x[0]
        if (x[1::] - x[0:-1] != delx).any():
            raise(ValueError, "pspec only defined on regular grid")
        s = np.fft.fftshift(np.fft.fftfreq(N, d=delx))
        return np.sqrt(2*np.pi*l)*sigmaf**2.0*np.exp(-l**2*s**2/2.0)
    else:
        xxp = abs_diff(x, xp)
        return sigmaf**2*np.exp(-xxp**2/(2*l**2))
