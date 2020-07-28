# -*- coding: utf-8 -*-


import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from africanus.model.wsclean.file_model import load
from africanus.model.wsclean.spec_model import (ordinary_spectral_model,
                                                log_spectral_model, spectra)
from africanus.model.wsclean.dask import spectra as dask_spectra


@pytest.fixture
def freq():
    return np.linspace(.856e9, .856e9*2, 16)


@pytest.fixture
def spectral_model_inputs(wsclean_model_file):
    sources = dict(load(wsclean_model_file))

    I, spi, log_si, ref_freq = (sources[n] for n in ("I", "SpectralIndex",
                                                     "LogarithmicSI",
                                                     "ReferenceFrequency"))

    I = np.asarray(I)  # noqa
    spi = np.asarray(spi)
    log_si = np.asarray(log_si)
    ref_freq = np.asarray(ref_freq)

    return I, spi, log_si, ref_freq


def test_spectral_model(spectral_model_inputs, freq):
    I, spi, log_si, ref_freq = spectral_model_inputs

    # Ensure positive flux for logarithmic polynomials
    I[log_si] = np.abs(I[log_si])
    spi[log_si] = np.abs(spi[log_si])

    # Compute spectral model with numpy implementations
    ordinary_spec_model = ordinary_spectral_model(I, spi, log_si,
                                                  freq, ref_freq)
    log_spec_model = log_spectral_model(I, spi, log_si,
                                        freq, ref_freq)

    # Choose between ordinary and log spectral index
    # based on log_si array
    spec_model = np.where(log_si[:, None] == True,  # noqa
                          log_spec_model,
                          ordinary_spec_model)

    # Compare with our implementation
    model = spectra(I, spi, log_si, ref_freq, freq)
    assert_array_almost_equal(model, spec_model)

    model = spectra(I, spi, False, ref_freq, freq)
    assert_array_almost_equal(model, ordinary_spec_model)

    log_si_all_true = np.full(I.shape, True, dtype=np.bool)
    I = np.abs(I)  # noqa
    spi = np.abs(spi)
    log_spec_model = log_spectral_model(I, spi, log_si_all_true,
                                        freq, ref_freq)
    model = spectra(I, spi, True, ref_freq, freq)
    assert_array_almost_equal(model, log_spec_model)


def test_dask_spectral_model(spectral_model_inputs, freq):
    da = pytest.importorskip("dask.array")

    I, spi, log_si, ref_freq = spectral_model_inputs

    # Ensure positive flux for logarithmic polynomials
    I[log_si] = np.abs(I[log_si])
    spi[log_si] = np.abs(spi[log_si])

    # Compute spectral model with numpy implementations
    ordinary_spec_model = ordinary_spectral_model(I, spi, log_si,
                                                  freq, ref_freq)
    log_spec_model = log_spectral_model(I, spi, log_si,
                                        freq, ref_freq)

    # Choose between ordinary and log spectral index
    # based on log_si array
    spec_model = np.where(log_si[:, None] == True,  # noqa
                          log_spec_model,
                          ordinary_spec_model)

    # Create dask arrays
    src_chunks = (4, 3)
    spi_chunks = (2,)
    freq_chunks = (4, 4, 4, 4)

    I = da.from_array(I, chunks=(src_chunks,))  # noqa
    spi = da.from_array(spi, chunks=(src_chunks, spi_chunks))
    log_si = da.from_array(log_si, chunks=(src_chunks,))
    ref_freq = da.from_array(ref_freq, chunks=(src_chunks,))
    freq = da.from_array(freq, chunks=(freq_chunks,))

    # Compute spectra and compare
    model = dask_spectra(I, spi, log_si, ref_freq, freq)
    assert_array_almost_equal(model, spec_model)
