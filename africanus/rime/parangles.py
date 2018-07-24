# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from ..util.requirements import have_packages, MissingPackageException
from ..util.docs import on_rtd

_discovered_backends = []
_standard_backends = ['casa', 'astropy']

_casa_requirements = ('pyrap.measures', 'pyrap.quanta')
have_casa_requirements = have_packages(*_casa_requirements)

if not have_casa_requirements or on_rtd():
    def casa_parallactic_angles(times, antenna_positions, field_centre):
        raise MissingPackageException(*_casa_requirements)
else:
    _discovered_backends.append('casa')

    import pyrap.measures
    import pyrap.quanta as pq

    # Create a measures server
    meas_serv = pyrap.measures.measures()

    def casa_parallactic_angles(times, antenna_positions, field_centre):
        """
        Computes parallactic angles per timestep for the given
        reference antenna position and field centre.
        """

        # Create direction measure for the zenith
        zenith = meas_serv.direction('AZELGEO', '0deg', '90deg')

        # Create position measures for each antenna
        reference_positions = [meas_serv.position(
                                    'itrf',
                                    *(pq.quantity(x, 'm') for x in pos))
                               for pos in antenna_positions]

        # Compute field centre in radians
        fc_rad = meas_serv.direction('J2000', *(pq.quantity(f, 'rad')
                                                for f in field_centre))

        return np.asarray([
            # Set current time as the reference frame
            meas_serv.do_frame(meas_serv.epoch("UTC", pq.quantity(t, "s")))
            and
            [   # Set antenna position as the reference frame
                meas_serv.do_frame(rp)
                and
                meas_serv.posangle(fc_rad, zenith).get_value("rad")
                for rp in reference_positions
            ]
            for t in times])

_astropy_requirements = ('astropy',)
have_astropy_requirements = have_packages(*_astropy_requirements)

if not have_astropy_requirements or on_rtd():
    def astropy_parallactic_angles(times, antenna_positions, field_centre):
        raise MissingPackageException(*_astropy_requirements)
else:
    _discovered_backends.append('astropy')

    from astropy.coordinates import (EarthLocation, SkyCoord,
                                     AltAz, CIRS)
    from astropy.time import Time
    from astropy import units

    def astropy_parallactic_angles(times, antenna_positions, field_centre):
        """
        Computes parallactic angles per timestep for the given
        reference antenna position and field centre.
        """
        ap = antenna_positions
        fc = field_centre

        # Convert from MJD second to MJD
        times = Time(times / 86400.00, format='mjd', scale='utc')

        ap = EarthLocation.from_geocentric(
            ap[:, 0], ap[:, 1], ap[:, 2], unit='m')
        fc = SkyCoord(ra=fc[0], dec=fc[1], unit=units.rad, frame='fk5')
        pole = SkyCoord(ra=0, dec=90, unit=units.deg, frame='fk5')

        cirs_frame = CIRS(obstime=times)
        pole_cirs = pole.transform_to(cirs_frame)
        fc_cirs = fc.transform_to(cirs_frame)

        altaz_frame = AltAz(location=ap[None, :], obstime=times[:, None])
        pole_altaz = pole_cirs[:, None].transform_to(altaz_frame)
        fc_altaz = fc_cirs[:, None].transform_to(altaz_frame)
        return fc_altaz.position_angle(pole_altaz)


def parallactic_angles(times, antenna_positions, field_centre, **kwargs):
    """
    Computes parallactic angles per timestep for the given
    reference antenna position and field centre.

    Notes
    -----

    * The python-casacore backend uses an ``AZELGEO`` frame in order
      to more closely agree with the astropy backend,
      but slightly differs from ``AZEL`` frame using in MeqTrees.
    * The astropy backend is slightly more than 2x faster than
      the casa backend
    * The astropy and python-casacore differ by at most
      10 arcseconds

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
    backend : {'casa', 'astropy', 'test'}, optional
        Backend to use for calculating the parallactic angles.

        * ``casa`` defers to an implementation
          depending on ``python-casacore``
        * ``astropy`` defers to astropy.
        * ``test`` creates parallactic angles
          by multiplying the ``times`` and ``antenna_position``
          arrays. It exist solely for testing.

    Returns
    -------
    :class:`numpy.ndarray`
        Parallactic angles of shape :code:`(time,ant)`
    """

    try:
        backend = kwargs.pop('backend')
    except KeyError:
        try:
            backend = _discovered_backends[0]
        except IndexError:
            raise ValueError("None of the standard backends "
                             "%s are installed" % _standard_backends)

    if not field_centre.shape == (2,):
        raise ValueError("Invalid field_centre shape %s" %
                         (field_centre.shape,))

    if backend == 'astropy':
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
