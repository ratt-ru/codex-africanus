#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import pytest


@pytest.fixture
def fits_header():
    return {
        "SIMPLE":                  'T',  # / conforms to FITS standard
        "BITPIX": -64,  # / array data type
        "NAXIS":                     3,  # / number of array dimensions
        "NAXIS1":                  513,
        "NAXIS2":                  513,
        "NAXIS3":                   33,
        "EXTEND":                  'T',
        "DATE":    '2015-05-20 12:40:12.507624',
        "DATE-OB": '2015-05-20 12:40:12.507624',
        "ORIGIN":  'SOMEONE ',
        "TELESCO": 'VLA     ',
        "OBJECT":  'beam    ',
        "EQUINOX":               2000.0,
        "CTYPE1": 'L       ',           # points right on the sky
        "CUNIT1": 'DEG     ',
        "CDELT1":             0.011082,  # degrees
        "CRPIX1":                  257,  # reference pixel (one relative)
        "CRVAL1":      0.0110828777007,
        "CTYPE2": '-M      ',           # points up on the sky
        "CUNIT2": 'DEG     ',
        "CDELT2":             0.011082,  # degrees
        "CRPIX2":                  257,  # reference pixel (one relative)
        "CRVAL2": -2.14349358381E-07,
        "CTYPE3": 'FREQ    ',
        "CDELT3":            1008000.0,  # frequency step in Hz
        "CRPIX3":                    1,  # reference frequency postion
        "CRVAL3":         1400256000.0,  # reference frequency
        "CTYPE4": 'STOKES  ',
        "CDELT4":                    1,
        "CRPIX4":                    1,
        "CRVAL4": -5,
        "GFREQ1":         1400256000.0,
        "GFREQ2":    1401267006.481463,
        "GFREQ3":    1402322911.080775,
        "GFREQ4":    1403413869.993157,
        "GFREQ5":    1404446534.122004,
        "GFREQ6":    1405431839.039557,
        "GFREQ7":    1406450580.210605,
        "GFREQ8":    1407565986.781461,
        "GFREQ9":    1408540601.110557,
        "GFREQ10":    1409590690.509872,
        "GFREQ11":    1410635261.125197,
        "GFREQ12":    1411713397.984036,
        "GFREQ13":    1412731853.361315,
        "GFREQ14":    1413826544.202757,
        "GFREQ15":     1414823303.16869,
        "GFREQ16":    1415817968.786441,
        "GFREQ17":    1416889091.051286,
        "GFREQ18":    1417937927.157403,
        "GFREQ19":    1419010194.848117,
        "GFREQ20":    1420027703.693506,
        "GFREQ21":    1421107695.319375,
        "GFREQ22":     1422148567.69773,
        "GFREQ23":    1423184370.515572,
        "GFREQ24":    1424165878.168865,
        "GFREQ25":    1425208894.904767,
        "GFREQ26":    1426298839.860366,
        "GFREQ27":    1427265196.336215,
        "GFREQ28":    1428354727.177189,
        "GFREQ29":    1429435689.132821,
        "GFREQ30":     1430380674.10678,
        "GFREQ31":    1431456384.211675,
        "GFREQ32":         1432512000.0,
        "GFREQ33":         1432456789.0,  # Last GFREQ hard-coded to
                                          # something non-linear
    }


def test_fits_axes(fits_header):
    from africanus.util.beams import BeamAxes

    beam_axes = BeamAxes(fits_header)

    # L axis converted to radian
    assert beam_axes.ctype[0] == fits_header['CTYPE1'].strip() == 'L'
    assert fits_header['CUNIT1'].strip() == 'DEG'
    assert beam_axes.cunit[0] == 'RAD'
    assert beam_axes.crval[0] == np.deg2rad(fits_header['CRVAL1'])
    assert beam_axes.cdelt[0] == np.deg2rad(fits_header['CDELT1'])
    assert beam_axes.sign[0] == 1.0

    # M axis converted to radian and sign flipped
    assert fits_header['CTYPE2'].strip() == '-M'
    assert beam_axes.ctype[1] == 'M'
    assert fits_header['CUNIT2'].strip() == 'DEG'
    assert beam_axes.cunit[1] == 'RAD'
    assert beam_axes.crval[1] == np.deg2rad(fits_header['CRVAL2'])
    assert beam_axes.cdelt[1] == np.deg2rad(fits_header['CDELT2'])
    assert beam_axes.sign[1] == -1.0

    # GFREQS used for the frequency grid
    gfreqs = [fits_header.get('GFREQ%d' % (i+1)) for i
              in range(fits_header['NAXIS3'])]

    assert np.allclose(beam_axes.grid[2], np.asarray(gfreqs))

    # Now remove a GFREQ, forcing usage of the regular frequency grid
    fits_header = fits_header.copy()
    del fits_header['GFREQ30']

    beam_axes = BeamAxes(fits_header)

    R = np.arange(beam_axes.naxis[2])
    g = (R - beam_axes.crpix[2])*beam_axes.cdelt[2] + beam_axes.crval[2]

    assert np.all(g == beam_axes.grid[2])


def test_beam_grids(fits_header):
    from africanus.util.beams import beam_grids

    hdr = fits_header

    # Extract l, m and frequency axes and grids
    (l, l_grid), (m, m_grid), (freq, freq_grid) = beam_grids(fits_header)

    # Check expected L
    crval = hdr['CRVAL%d' % l]
    cdelt = hdr['CDELT%d' % l]
    crpix = hdr['CRPIX%d' % l] - 1  # C-indexing
    R = np.arange(0.0, float(hdr['NAXIS%d' % l]))

    exp_l = (R - crpix)*cdelt + crval
    exp_l = np.deg2rad(exp_l)

    assert np.allclose(exp_l, l_grid)

    crval = hdr['CRVAL%d' % m]
    cdelt = hdr['CDELT%d' % m]
    crpix = hdr['CRPIX%d' % m] - 1  # C-indexing
    R = np.arange(0.0, float(hdr['NAXIS%d' % m]))

    # Check expected M. It's -M in the FITS header
    # so there's a flip in direction here
    exp_m = (R - crpix)*cdelt + crval
    exp_m = np.deg2rad(exp_m)
    exp_m = np.flipud(exp_m)

    assert np.allclose(exp_m, m_grid)

    # GFREQS used for the frequency grid
    gfreqs = [fits_header.get('GFREQ%d' % (i+1)) for i
              in range(fits_header['NAXIS3'])]

    assert np.allclose(freq_grid, gfreqs)


def test_beam_filenames():
    from africanus.util.beams import beam_filenames

    assert beam_filenames("beam_$(corr)_$(reim).fits", [9, 10, 11, 12]) == {
        'xx': ['beam_xx_re.fits', 'beam_xx_im.fits'],
        'xy': ['beam_xy_re.fits', 'beam_xy_im.fits'],
        'yx': ['beam_yx_re.fits', 'beam_yx_im.fits'],
        'yy': ['beam_yy_re.fits', 'beam_yy_im.fits']
    }

    assert beam_filenames("beam_$(corr)_$(reim).fits", [5, 6, 7, 8]) == {
        'rr': ['beam_rr_re.fits', 'beam_rr_im.fits'],
        'rl': ['beam_rl_re.fits', 'beam_rl_im.fits'],
        'lr': ['beam_lr_re.fits', 'beam_lr_im.fits'],
        'll': ['beam_ll_re.fits', 'beam_ll_im.fits']
    }

    assert beam_filenames("beam_$(CORR)_$(reim).fits", [9, 10, 11, 12]) == {
        'xx': ['beam_XX_re.fits', 'beam_XX_im.fits'],
        'xy': ['beam_XY_re.fits', 'beam_XY_im.fits'],
        'yx': ['beam_YX_re.fits', 'beam_YX_im.fits'],
        'yy': ['beam_YY_re.fits', 'beam_YY_im.fits']
    }

    assert beam_filenames("beam_$(corr)_$(REIM).fits", [9, 10, 11, 12]) == {
        'xx': ['beam_xx_RE.fits', 'beam_xx_IM.fits'],
        'xy': ['beam_xy_RE.fits', 'beam_xy_IM.fits'],
        'yx': ['beam_yx_RE.fits', 'beam_yx_IM.fits'],
        'yy': ['beam_yy_RE.fits', 'beam_yy_IM.fits']
    }


def test_inverse_interp():
    """
    Tests that interp1d handles monotically increasing
    and decreasing correctly
    """

    try:
        from scipy.interpolate import interp1d
    except ImportError:
        pytest.skip("scipy not installed")

    # Monotically decreasing
    values = np.asarray([1.0, 0.7, 0.2, 0.0, -0.4, -1.0])
    assert np.all(np.diff(values) < 0)
    grid = np.arange(values.size)

    initial = np.stack((values, grid))
    interp = interp1d(values, grid, bounds_error=False,
                      fill_value='extrapolate')
    assert np.all(initial == np.stack((values, interp(values))))

    # Monotonically increasing
    values = np.flipud(values)
    assert np.all(np.diff(values) > 0)
    initial = np.stack((values, grid))
    interp = interp1d(values, grid, bounds_error=False,
                      fill_value='extrapolate')
    assert np.all(initial == np.stack((values, interp(values))))
