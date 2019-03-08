# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from africanus.model import spectra
from africanus.model.dask import spectra as dask_spectra
from africanus.model.apps.wsclean_file_model import wsclean


@pytest.fixture
def freq():
    return np.linspace(.856e9, .856e9*2, 16)


@pytest.fixture
def spectral_model_inputs(wsclean_model_file):
    sources = wsclean(wsclean_model_file)

    _, _, _, _, I, spi, log_si, ref_freq, _, _, _ = sources

    I = np.asarray(I)  # noqa
    spi = np.asarray(spi)
    log_si = np.asarray(log_si)
    ref_freq = np.asarray(ref_freq)

    return I, spi, log_si, ref_freq


def ordinary_spectral_model(I, spi, log_si, freq, ref_freq):
    spi_idx = np.arange(1, spi.shape[1] + 1)
    # (source, chan, spi-comp)
    term = (freq[None, :, None] / ref_freq[:, None, None]) - 1.0
    term = term**spi_idx[None, None, :]
    term = spi[:, None, :]*term
    return I[:, None] + term.sum(axis=2)


def log_spectral_model(I, spi, log_si, freq, ref_freq):
    # No negative flux
    I = np.where(log_si == False, 1.0, I)  # noqa
    spi_idx = np.arange(1, spi.shape[1] + 1)
    # (source, chan, spi-comp)
    term = np.log(freq[None, :, None] / ref_freq[:, None, None])
    term = term**spi_idx[None, None, :]
    term = spi[:, None, :]*term
    return np.exp(np.log(I)[:, None] + term.sum(axis=2))


def test_spectral_model(spectral_model_inputs, freq):
    I, spi, log_si, ref_freq = spectral_model_inputs

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

    # Ensure positive flux
    posI = I.copy()
    posI[I < 0.0] = -posI[I < 0.0]
    log_si_all_true = np.full(posI.shape, True, dtype=np.bool)
    log_spec_model = log_spectral_model(posI, spi, log_si_all_true,
                                        freq, ref_freq)
    model = spectra(posI, spi, True, ref_freq, freq)
    assert_array_almost_equal(model, log_spec_model)

    model = spectra(I, spi, False, ref_freq, freq)
    assert_array_almost_equal(model, ordinary_spec_model)


def _broadcast_corrs(array, corrs):
    idx = (tuple(slice(None) for _ in array.shape) +
           tuple(None for _ in corrs))

    shape = array.shape + corrs
    broadcast_array = np.broadcast_to(array[idx], shape)

    return np.require(broadcast_array, requirements=["C"])


@pytest.mark.parametrize("corrs", [(), (1,), (2,), (2, 2)])
def test_spectral_model_corrs(spectral_model_inputs, freq, corrs):
    I, spi, log_si, ref_freq = spectral_model_inputs

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

    # Just broadcast everything up to test
    I = _broadcast_corrs(I, corrs)  # noqa
    spi = _broadcast_corrs(spi, corrs)
    log_si = _broadcast_corrs(log_si, corrs)
    spec_model = _broadcast_corrs(spec_model, corrs)

    model = spectra(I, spi, log_si, ref_freq, freq)
    assert_array_almost_equal(model, spec_model)


@pytest.mark.parametrize("corrs", [(), (1,), (2,), (2, 2)])
def test_dask_spectral_model(spectral_model_inputs, freq, corrs):
    da = pytest.importorskip("dask.array")

    I, spi, log_si, ref_freq = spectral_model_inputs

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

    # Just broadcast everything up to test
    I = _broadcast_corrs(I, corrs)  # noqa
    spi = _broadcast_corrs(spi, corrs)
    log_si = _broadcast_corrs(log_si, corrs)
    spec_model = _broadcast_corrs(spec_model, corrs)

    # Create dask arrays
    src_chunks = (4, 3)
    spi_chunks = (2,)

    I = da.from_array(I, chunks=(src_chunks,) + corrs)  # noqa
    spi = da.from_array(spi, chunks=(src_chunks, spi_chunks) + corrs)
    log_si = da.from_array(log_si, chunks=(src_chunks,) + corrs)
    ref_freq = da.from_array(ref_freq, chunks=(src_chunks,))
    freq = da.from_array(freq, chunks=4)

    # Compute spectra and compare
    model = dask_spectra(I, spi, log_si, ref_freq, freq)
    assert_array_almost_equal(model, spec_model)
