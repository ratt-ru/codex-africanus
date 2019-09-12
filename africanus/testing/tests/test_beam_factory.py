# -*- coding: utf-8 -*-


from africanus.testing.beam_factory import beam_factory

import pytest


@pytest.mark.parametrize("pol_type", ["linear", "circular"])
def test_beam_factory(tmp_path, pol_type):
    fits = pytest.importorskip('astropy.io.fits')
    schema = tmp_path / "test_beam_$(corr)_$(reim).fits"

    filenames = beam_factory(schema=schema,
                             npix=15,
                             polarisation_type=pol_type)

    for corr, (re_file, im_file) in filenames.items():
        with fits.open(re_file), fits.open(im_file):
            pass
