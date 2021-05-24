# -*- coding: utf-8 -*-


import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from africanus.model.wsclean.file_model import load, arcsec2rad


def test_wsclean_model_file(wsclean_model_file):
    sources = dict(load(wsclean_model_file))

    (name, stype, ra, dec, I,
     spi, log_si, ref_freq,
     major, minor, orientation) = (sources[n] for n in (
                                   "Name", "Type", "Ra", "Dec", "I",
                                   "SpectralIndex", "LogarithmicSI",
                                   "ReferenceFrequency",
                                   "MajorAxis", "MinorAxis", "Orientation"))

    # Seven sources
    assert (len(I) == len(spi) == len(log_si) == len(ref_freq) == 8)

    # Name and type read correctly
    assert name[0] == "s0c0" and stype[0] == "POINT"

    # Check ra conversion for line 0 file entry (-float, float, float)
    hours, mins, secs = (-8., 28., 5.152)
    expected_ra0 = -2.0 * np.pi * (
                    (-hours / 24.0) +
                    (mins / (24.0*60.0)) +
                    (secs / (24.0*60.0*60.0)))

    assert ra[0] == expected_ra0

    # Check dec conversion for line 0 file entry
    degs, mins, secs = (39., 35., 8.511)
    expected_dec0 = 2.0 * np.pi * (
                     (degs / 360.0) +
                     (mins / (360.0*60.0)) +
                     (secs / (360.0*60.0*60.0)))

    assert dec[0] == expected_dec0

    # SPI read correctly
    assert_array_equal(spi[0], [-0.00695379313004673, -0.0849693907803257])

    # LogrithmicSI read correctly
    assert log_si[0] is True

    # Check ra conversion for line 2 file entry (int, not float, seconds)
    hours, mins, secs = (8, 18, 44)
    expected_ra2 = 2.0 * np.pi * (
                    (hours / 24.0) +
                    (mins / (24.0*60.0)) +
                    (secs / (24.0*60.0*60.0)))

    assert ra[2] == expected_ra2

    # Check dec conversion for line 2 file entry (int, not float, seconds)
    degs, mins, secs = (39, 38, 37)
    expected_dec2 = 2.0 * np.pi * (
                     (degs / 360.0) +
                     (mins / (360.0*60.0)) +
                     (secs / (360.0*60.0*60.0)))

    assert dec[2] == expected_dec2

    assert log_si[2] is False

    assert I[2] == 0.000233552686127518

    # Check dec conversion for line 4 file entry (+int, not float, seconds)
    degs, mins, secs = (+41, 47, 17.131)
    expected_dec4 = 2.0 * np.pi * (
                     (degs / 360.0) +
                     (mins / (360.0*60.0)) +
                     (secs / (360.0*60.0*60.0)))

    assert dec[4] == expected_dec4

    # Missing reference frequency set in the last
    assert ref_freq[6] == ref_freq[0]

    # Last name and type correct
    assert name[6] == "s1c2" and stype[6] == "GAUSSIAN"

    # https://www.convertunits.com/from/arcsecond/to/radian
    assert_array_almost_equal(major[6], arcsec2rad(83.6144111272856))
    assert_array_almost_equal(minor[6], arcsec2rad(83.6144111272856))
    assert_array_almost_equal(orientation[6], np.deg2rad(45))

    assert I[6] == 0.000660490865128381

    assert I[7] == 0
    assert_array_equal(spi[7], [0.0, 0.0])
