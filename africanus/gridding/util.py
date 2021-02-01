
import numpy as np


def estimate_cell_size(u, v, wavelength, factor=3.0, ny=None, nx=None):
    r"""
    Estimate the cell size in arcseconds given
    baseline ``u`` and ``v`` coordinates, as well
    as the ``wavelengths``, :math:`\lambda`.

    The cell size is computed as:

    .. math::

        \Delta u = 1.0 / \left( 2 \times \text{ factor }
                                  \times \max (\vert u \vert)
                                  / \min( \lambda) \right)

        \Delta v = 1.0 / \left( 2 \times \text{ factor }
                                  \times \max (\vert v \vert)
                                  / \min( \lambda) \right)


    If ``ny`` and ``nx`` are provided the following checks are performed
    and exceptions are raised on failure:

    .. math::

        \Delta u * \text{ ny } \leq \min (\lambda) / \min (\vert u \vert)

        \Delta v * \text{ nx } \leq \min (\lambda) / \min (\vert v \vert)

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
        abs_u = np.abs(u)
        umax = abs_u.max()
        umin = abs_u.min()
    elif isinstance(u, float):
        umax = umin = abs(u)
    else:
        raise TypeError("Invalid u type %s" % type(u))

    if isinstance(v, np.ndarray):
        abs_v = np.abs(v)
        vmax = abs_v.max()
        vmin = abs_v.min()
    elif isinstance(v, float):
        vmax = vmin = abs(v)
    else:
        raise TypeError("Invalid v type %s" % type(v))

    if isinstance(wavelength, np.ndarray):
        wave_min = wavelength.min()
    elif isinstance(wavelength, float):
        wave_min = wavelength
    else:
        raise TypeError("Invalid wavelength type %s" % type(v))

    umax /= wave_min
    vmax /= wave_min
    umin /= wave_min
    vmin /= wave_min

    u_cell_size = 1.0 / (2.0 * factor * umax)
    v_cell_size = 1.0 / (2.0 * factor * vmax)

    if ny is not None and u_cell_size*ny < (1.0 / umin):
        raise ValueError("v_cell_size*ny [%f] < (1.0 / umin) [%f]" %
                         (u_cell_size*ny, 1.0 / umin))

    if nx is not None and v_cell_size*nx < (1.0 / vmin):
        raise ValueError("v_cell_size*nx [%f] < (1.0 / vmin) [%f]" %
                         (v_cell_size*nx, 1.0 / vmin))

    # Convert radians to arcseconds
    return np.rad2deg([u_cell_size, v_cell_size])*(60*60)
