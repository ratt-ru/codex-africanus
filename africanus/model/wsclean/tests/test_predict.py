# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_almost_equal

from africanus.rime import phase_delay
from africanus.model.coherency import convert
from africanus.model.wsclean.spec_model import spectra
from africanus.model.wsclean.predict import predict


def test_wsclean_predict():
    row = 10
    src = 10
    chan = 16

    rs = np.random.RandomState(42)
    uvw = rs.normal(size=(row, 3))
    lm = rs.normal(size=(src, 2))*1e-5
    flux = rs.normal(size=src)
    coeffs = rs.normal(size=(src, 2))
    log_poly = rs.randint(0, 1, src, dtype=np.bool)
    freq = np.linspace(.856e9, 2*.856e9, chan)
    ref_freq = np.full(src, freq[freq.shape[0] // 2])

    # WSClean visibilities
    vis = predict(uvw, lm, flux, coeffs, log_poly, freq, ref_freq)

    # Compute it another way.
    # Note the inverted UVW coordinates to get appropriate sign convention
    phase = phase_delay(lm, -uvw, freq)
    spectrum = spectra(flux, coeffs, log_poly, ref_freq, freq)
    np_vis = np.einsum("srf,sf->rf", phase, spectrum)[:, :, None]

    assert_almost_equal(np_vis, vis)
