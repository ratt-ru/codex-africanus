# -*- coding: utf-8 -*-


import warnings

from .parangles_astropy import (have_astropy_parangles,
                                astropy_parallactic_angles)
from .parangles_casa import (have_casa_parangles,
                             casa_parallactic_angles)

_discovered_backends = []

if have_astropy_parangles:
    _discovered_backends.append('astropy')

if have_casa_parangles:
    _discovered_backends.append('casa')


_standard_backends = set(['casa', 'astropy', 'test'])


def parallactic_angles(times, antenna_positions, field_centre,
                       backend='casa'):
    """
    Computes parallactic angles per timestep for the given
    reference antenna position and field centre.

    Parameters
    ----------
    times : :class:`numpy.ndarray`
        Array of Mean Julian Date times in
        *seconds* with shape :code:`(time,)`,
    antenna_positions : :class:`numpy.ndarray`
        Antenna positions of shape :code:`(ant, 3)`
        in *metres* in the *ITRF* frame.
    field_centre : :class:`numpy.ndarray`
        Field centre of shape :code:`(2,)` in *radians*
    backend : {'casa', 'test'}, optional
        Backend to use for calculating the parallactic angles.

        * ``casa`` defers to an implementation
          depending on ``python-casacore``.
          This backend should be used by default.
        * ``test`` creates parallactic angles
          by multiplying the ``times`` and ``antenna_position``
          arrays. It exist solely for testing.

    Returns
    -------
    parallactic_angles : :class:`numpy.ndarray`
        Parallactic angles of shape :code:`(time,ant)`
    """
    if backend not in _standard_backends:
        raise ValueError("'%s' is not one of the "
                         "standard backends '%s'"
                         % (backend, _standard_backends))

    if not field_centre.shape == (2,):
        raise ValueError("Invalid field_centre shape %s" %
                         (field_centre.shape,))

    if backend == 'astropy':
        warnings.warn('astropy backend currently returns the incorrect values')
        return astropy_parallactic_angles(times,
                                          antenna_positions,
                                          field_centre)
    elif backend == 'casa':
        return casa_parallactic_angles(times,
                                       antenna_positions,
                                       field_centre)
    elif backend == 'test':
        return times[:, None]*(antenna_positions.sum(axis=1)[None, :])
    else:
        raise ValueError("Invalid backend %s" % backend)
