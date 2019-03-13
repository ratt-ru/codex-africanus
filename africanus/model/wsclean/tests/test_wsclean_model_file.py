# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from africanus.model.wsclean.file_model import load


def test_wsclean_model_file(wsclean_model_file):
    sources = dict(load(wsclean_model_file))

    (name, stype, ra, dec, I,
     spi, log_si, ref_freq) = (sources[n] for n in (
                                            "Name", "Type", "Ra", "Dec", "I",
                                            "SpectralIndex", "LogarithmicSI",
                                            "ReferenceFrequency"))

    # Seven sources
    assert (len(I) == len(spi) == len(log_si) == len(ref_freq) == 7)

    # Name and type read correctly
    assert name[0] == "s0c0" and stype[0] == "POINT"

    # Check ra conversion for line 0 file entry
    hours, mins, secs = (-8., 28., 5.152)
    expected_ra0 = 2.0 * np.pi * (
                    (-hours / 24.0) +
                    (mins / (24.0*60.0)) +
                    (secs / (24.0*60.0*60.0)))

    assert ra[0] == expected_ra0

    # Check dec conversion for line 0 file entry
    degs, mins, secs = (39., 35., 8.511)
    expected_dec0 = 2.0 * np.pi * (
                     (degs / 360.0) -
                     (mins / (360.0*60.0)) -
                     (secs / (360.0*60.0*60.0)))

    assert dec[0] == expected_dec0

    # SPI read correctly
    assert spi[0] == [-0.00695379313004673, -0.0849693907803257]

    # LogrithmicSI read correctly
    assert log_si[0] is True

    # Check ra conversion for line 2 file entry (int, not float, seconds)
    hours, mins, secs = (8, 18, 44)
    expected_ra2 = 2.0 * np.pi * (
                    (hours / 24.0) -
                    (mins / (24.0*60.0)) -
                    (secs / (24.0*60.0*60.0)))

    assert ra[2] == expected_ra2

    # Check dec conversion for line 0 file entry (int, not float, seconds)
    degs, mins, secs = (39, 38, 37)
    expected_dec2 = 2.0 * np.pi * (
                     (degs / 360.0) -
                     (mins / (360.0*60.0)) -
                     (secs / (360.0*60.0*60.0)))

    assert dec[2] == expected_dec2

    assert log_si[2] is False

    assert I[2] == 0.000233552686127518

    # Missing reference frequency set in the last
    assert ref_freq[-1] == ref_freq[0]

    # Last name and type correct
    assert name[-1] == "s1c2" and stype[-1] == "GAUSSIAN"

    assert I[-1] == 0.000660490865128381
