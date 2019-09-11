# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from africanus.testing.beam_factory import beam_factory

import pytest


def test_beam_factory(tmp_path):
    fits = pytest.importorskip('astropy.io.fits')
    schema = tmp_path / "test_beam_$(corr)_$(reim).fits"

    filenames = beam_factory(schema=schema)

    for corr, (re, im) in filenames.items():
        with fits.open(re), fits.open(im):
            pass
