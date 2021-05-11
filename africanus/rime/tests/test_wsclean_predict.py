# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from africanus.rime import phase_delay
from africanus.model.shape.gaussian_shape import gaussian
from africanus.model.wsclean.spec_model import spectra
from africanus.rime.wsclean_predict import wsclean_predict

chunk_parametrization = pytest.mark.parametrize("chunks", [
    {
        'source':  (2, 3, 4, 2, 2, 2, 2, 2, 2),
        'time': (2, 1, 1),
        'rows': (4, 4, 2),
        'antenna': (4,),
        'channels': (3, 2),
    }])


@chunk_parametrization
def test_wsclean_predict(chunks):
    row = sum(chunks['rows'])
    src = sum(chunks['source'])
    chan = sum(chunks['channels'])

    rs = np.random.RandomState(42)
    source_sel = rs.randint(0, 2, src).astype(np.bool_)
    source_type = np.where(source_sel, "POINT", "GAUSSIAN")

    gauss_shape = rs.normal(size=(src, 3))
    uvw = rs.normal(size=(row, 3))
    lm = rs.normal(size=(src, 2))*1e-5
    flux = rs.normal(size=src)
    coeffs = rs.normal(size=(src, 2))
    log_poly = rs.randint(0, 2, src, dtype=np.bool_)
    flux[log_poly] = np.abs(flux[log_poly])
    coeffs[log_poly] = np.abs(coeffs[log_poly])
    freq = np.linspace(.856e9, 2*.856e9, chan)
    ref_freq = np.full(src, freq[freq.shape[0] // 2])

    # WSClean visibilities
    vis = wsclean_predict(uvw, lm, source_type, flux, coeffs,
                          log_poly, ref_freq, gauss_shape, freq)

    # Compute it another way. Note the CASA coordinate convention
    # used by wsclean
    phase = phase_delay(lm, uvw, freq, convention='casa')
    spectrum = spectra(flux, coeffs, log_poly, ref_freq, freq)
    shape = gaussian(uvw, freq, gauss_shape)
    # point sources don't' contribute to the shape
    shape[source_sel] = 1.0
    np_vis = np.einsum("srf,srf,sf->rf", shape, phase, spectrum)[:, :, None]

    assert_almost_equal(np_vis, vis)


@chunk_parametrization
def test_dask_wsclean_predict(chunks):
    da = pytest.importorskip("dask.array")

    from africanus.rime.dask_predict import (
            wsclean_predict as dask_wsclean_predict)

    row = sum(chunks['rows'])
    src = sum(chunks['source'])
    chan = sum(chunks['channels'])

    rs = np.random.RandomState(42)
    source_sel = rs.randint(0, 2, src).astype(np.bool_)
    source_type = np.where(source_sel, "POINT", "GAUSSIAN")

    gauss_shape = rs.normal(size=(src, 3))

    uvw = rs.normal(size=(row, 3))
    lm = rs.normal(size=(src, 2))*1e-5
    flux = rs.normal(size=src)
    coeffs = rs.normal(size=(src, 2))
    log_poly = rs.randint(0, 2, src, dtype=np.bool_)
    flux[log_poly] = np.abs(flux[log_poly])
    coeffs[log_poly] = np.abs(coeffs[log_poly])
    freq = np.linspace(.856e9, 2*.856e9, chan)
    ref_freq = np.full(src, freq[freq.shape[0] // 2])

    da_uvw = da.from_array(uvw, chunks=(chunks['rows'], 3))
    da_lm = da.from_array(lm, chunks=(chunks['source'], 2))
    da_source_type = da.from_array(source_type, chunks=chunks['source'])
    da_gauss_shape = da.from_array(gauss_shape, chunks=(chunks['source'], 3))
    da_flux = da.from_array(flux, chunks=chunks['source'])
    da_coeffs = da.from_array(coeffs, chunks=(chunks['source'], 2))
    da_log_poly = da.from_array(log_poly, chunks=chunks['source'])
    da_ref_freq = da.from_array(ref_freq, chunks=chunks['source'])
    da_freq = da.from_array(freq)

    vis = wsclean_predict(uvw, lm, source_type, flux,
                          coeffs, log_poly, ref_freq,
                          gauss_shape, freq)
    da_vis = dask_wsclean_predict(da_uvw, da_lm, da_source_type,
                                  da_flux, da_coeffs,
                                  da_log_poly, da_ref_freq,
                                  da_gauss_shape, da_freq)

    assert_almost_equal(vis, da_vis)
