from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def estimate_cell_size(u, v, wavelength, factor=3.0, ny=None, nx=None):
    r"""
    Estimate the cell size in arcseconds given the
    baseline ``u`` and ``v`` coordinates, as well
    as the ``wavelengths``.

    The cell size is computed as:

    .. math::

        \Delta u = 1.0 / \left( 2 \times \text{ factor }
                                  \times \max (\vert u \vert)
                                  / \text{ wavelength} \right)

        \Delta v = 1.0 / \left( 2 \times \text{ factor }
                                  \times \max (\vert v \vert)
                                  / \text{ wavelength} \right)


    If ``ny`` and ``nx`` are provided the following checks are performed
    and exceptions are raised on failure:

    .. math::

        \Delta u * \text{ ny } \leq \left(1.0 / \max (\vert u \vert) \right)

        \Delta v * \text{ nx } \leq \left(1.0 / \max (\vert v \vert) \right)

    Parameters
    ----------
    u : :class:`numpy.ndarray` or float
        Maximum ``u`` coordinate in metres.
    v : :class:`numpy.ndarray` or float
        Maximum ``v`` coordinate in metres.
    wavelength : :class:`numpy.ndarray` or float
        Wavelengths, in metres.
    factor : float, optional
        Scaling factor
    ny : int, optional
        Grid y dimension
    nx : int, optional
        Grid x dimension

    Raises
    ------
    ValueError
        If the cell size criteria are not matched.

    Returns
    -------
    :class:`numpy.ndarray`
        Cell size of ``u`` and ``v`` in arcseconds with shape :code:`(2,)`
    """
    if isinstance(u, np.ndarray):
        umax = np.abs(u).max()
    elif isinstance(u, float):
        umax = abs(u)
    else:
        raise TypeError("Invalid u type %s" % type(u))

    if isinstance(v, np.ndarray):
        vmax = np.abs(v).max()
    elif isinstance(v, float):
        vmax = abs(v)
    else:
        raise TypeError("Invalid v type %s" % type(v))

    if isinstance(wavelength, np.ndarray):
        wave_max = wavelength.max()
    elif isinstance(wavelength, float):
        wave_max = wavelength
    else:
        raise TypeError("Invalid wavelength type %s" % type(v))

    u_cell_size = 1.0 / (2.0 * factor * umax / wave_max)
    v_cell_size = 1.0 / (2.0 * factor * vmax / wave_max)

    if ny is not None and u_cell_size*ny < (1.0 / umax):
        raise ValueError("u_cell_size*ny < (1.0 / umax)")

    if nx is not None and v_cell_size*nx < (1.0 / vmax):
        raise ValueError("v_cell_size*nx < (1.0 / vmax)")

    # Convert radians to arcseconds
    return np.rad2deg([u_cell_size, v_cell_size])*(60*60)
