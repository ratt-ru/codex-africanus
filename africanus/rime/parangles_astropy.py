# -*- coding: utf-8 -*-


from africanus.util.requirements import requires_optional

try:
    from astropy import units
    from astropy.coordinates import CIRS, AltAz, EarthLocation, SkyCoord
    from astropy.time import Time
except ImportError as e:
    astropy_import_error = e
    have_astropy_parangles = False
else:
    astropy_import_error = None
    have_astropy_parangles = True


@requires_optional("astropy", astropy_import_error)
def astropy_parallactic_angles(times, antenna_positions, field_centre):
    """
    Computes parallactic angles per timestep for the given
    reference antenna position and field centre.

    The parallactic angle is the position angle of the
    Celestial Intermediate Pole (the true celestial pole of date),
    measured from the apparent topocentric field centre in
    each antenna's horizontal frame. This includes precession,
    nutation, annual and diurnal aberration, and IERS polar motion
    and UT1-UTC (when astropy's IERS tables are available).
    """
    ap = antenna_positions
    fc = field_centre

    # Convert from MJD second to MJD
    times = Time(times / 86400.00, format="mjd", scale="utc")

    ap = EarthLocation.from_geocentric(ap[:, 0], ap[:, 1], ap[:, 2], unit="m")
    fc = SkyCoord(ra=fc[0], dec=fc[1], unit=units.rad, frame="fk5")

    # The pole of the CIRS frame is the Celestial Intermediate Pole
    cirs_frame = CIRS(obstime=times[:, None], location=ap[None, :])
    pole = SkyCoord(ra=0, dec=90, unit=units.deg, frame=cirs_frame)

    altaz_frame = AltAz(location=ap[None, :], obstime=times[:, None])
    pole_altaz = pole.transform_to(altaz_frame)
    fc_altaz = fc.transform_to(altaz_frame)
    return fc_altaz.position_angle(pole_altaz)
