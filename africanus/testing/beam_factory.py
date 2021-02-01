# -*- coding: utf-8 -*-


from pathlib import Path

import numpy as np

from africanus.util.beams import beam_filenames
from africanus.util.requirements import requires_optional

try:
    from astropy.io import fits
except ImportError as e:
    opt_import_error = e
else:
    opt_import_error = None


DEFAULT_SCHEMA = Path("test_beam_$(corr)_$(reim).fits")
LINEAR_CORRELATIONS = [9, 10, 11, 12]
CIRCULAR_CORRELATIONS = [5, 6, 7, 8]


BITPIX_MAP = {
    np.int8: 8,
    np.uint8: 8,
    np.int16: 16,
    np.uint16: 16,
    np.int32: 32,
    np.uint32: 32,
    np.float32: -32,
    np.float64: -64,
}


@requires_optional('astropy.io.fits', opt_import_error)
def beam_factory(polarisation_type='linear',
                 frequency=None,
                 npix=257,
                 dtype=np.float64,
                 schema=DEFAULT_SCHEMA,
                 overwrite=True):
    """ Generate a MeqTrees compliant beam cube """

    if npix % 2 != 1:
        raise ValueError("npix '%d' must be odd" % npix)

    # MeerKAT l-band, 64 channels
    if frequency is None:
        frequency = np.linspace(.856e9, .856e9*2, 64,
                                endpoint=True, dtype=np.float64)

    # Generate a linear space of grid frequencies
    gfrequency = np.linspace(frequency[0], frequency[-1],
                             33, dtype=np.float64)
    bandwidth = gfrequency[-1] - frequency[0]
    bandwidth_delta = bandwidth / gfrequency.shape[0]-1

    if polarisation_type == 'linear':
        CORR = LINEAR_CORRELATIONS
    elif polarisation_type == 'circular':
        CORR = CIRCULAR_CORRELATIONS
    else:
        raise ValueError("Invalid polarisation_type %s" % polarisation_type)

    extent_deg = 3.0
    coords = np.linspace(-extent_deg/2, extent_deg/2, npix, endpoint=True)

    crpix = 1 + npix // 2            # Reference pixel (FORTRAN)
    crval = coords[crpix - 1]        # Reference value
    cdelt = extent_deg / (npix - 1)  # Delta

    # List of key values of the form:
    #
    #    (key, None)
    #    (key, (value,))
    #    (key, (value, comment))
    #
    # We put them in a list so that they are added to the
    # FITS header in the correct order
    axis1 = [
        ("CTYPE", ('X', "points right on the sky")),
        ("CUNIT", ('DEG', 'degrees')),
        ("NAXIS", (npix, "number of X")),
        ("CRPIX", (crpix, "reference pixel (one relative)")),
        ("CRVAL", (crval, "degrees")),
        ("CDELT", (cdelt, "degrees"))]

    axis2 = [
        ("CTYPE", ('Y', "points up on the sky")),
        ("CUNIT", ('DEG', 'degrees')),
        ("NAXIS", (npix, "number of Y")),
        ("CRPIX", (crpix, "reference pixel (one relative)")),
        ("CRVAL", (crval, "degrees")),
        ("CDELT", (cdelt, "degrees"))]

    axis3 = [
        ("CTYPE", ('FREQ', )),
        ("CUNIT", None),
        ("NAXIS", (gfrequency.shape[0], "number of FREQ")),
        ("CRPIX", (1, "reference frequency position")),
        ("CRVAL", (gfrequency[0], "reference frequency")),
        ("CDELT", (bandwidth_delta, "frequency step in Hz"))]

    axes = [axis1, axis2, axis3]

    metadata = [
        ('SIMPLE', True),
        ('BITPIX', BITPIX_MAP[dtype]),
        ('NAXIS', len(axes)),
        ('OBSERVER', "Astronomer McAstronomerFace"),
        ('ORIGIN', "Artificial"),
        ('TELESCOP', "Telescope"),
        ('OBJECT', 'beam'),
        ('EQUINOX', 2000.0),
    ]

    # Create header and set metadata
    header = fits.Header(metadata)

    # Now set the key value entries for each axis
    ax_info = [('%s%d' % (k, a),) + vt
               for a, axis_data in enumerate(axes, 1)
               for k, vt in axis_data
               if vt is not None]
    header.update(ax_info)

    # Now setup the GFREQS
    # Jitter them randomly, except for the endpoints
    frequency_jitter = np.random.random(size=gfrequency.shape)-0.5
    frequency_jitter *= 0.1*bandwidth_delta
    frequency_jitter[0] = frequency_jitter[-1] = 0.0
    gfrequency += frequency_jitter

    # Check that gfrequency is monotically increasing
    assert np.all(np.diff(gfrequency) >= 0.0)

    for i, gfreq in enumerate(gfrequency, 1):
        header['GFREQ%d' % i] = gfreq

    # Figure out the beam filenames from the schema
    filenames = beam_filenames(str(schema), CORR)

    # Westerbork beam model
    coords = np.deg2rad(coords)
    r = np.sqrt(coords[None, :, None]**2 + coords[None, None, :]**2)
    fq = gfrequency[:, None, None]
    beam = np.cos(np.minimum(65*fq*1e-9*r, 1.0881))**3

    for filename in [f for ri_pair in filenames.values() for f in ri_pair]:
        primary_hdu = fits.PrimaryHDU(beam, header=header)
        primary_hdu.writeto(filename, overwrite=overwrite)

    return filenames


if __name__ == "__main__":
    beam_factory()
