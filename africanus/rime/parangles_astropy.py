# -*- coding: utf-8 -*-


from africanus.util.requirements import requires_optional

try:
    from astropy.coordinates import (EarthLocation, SkyCoord,
                                     AltAz, CIRS)
    from astropy.time import Time
    from astropy import units
except ImportError as e:
    astropy_import_error = e
    have_astropy_parangles = False
else:
    astropy_import_error = None
    have_astropy_parangles = True


@requires_optional('astropy', astropy_import_error)
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
