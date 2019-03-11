# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from africanus.model.spec_model import spectral_model, numpy_spectral_model


@pytest.fixture
def flux():
    def impl(nsrc):
        return np.random.random(nsrc)

    return impl


@pytest.fixture
def frequency():
    def impl(nchan):
        return np.linspace(.856e9, 2*.856e9, nchan)

    return impl


@pytest.fixture
def ref_freq():
    def impl(shape):
        return np.full(shape, 3*.856e9/2)

    return impl


@pytest.fixture
def spi():
    def impl(shape):
        return 0.7 + np.random.random(shape) * 0.2

    return impl


def test_spectral_model_one_spi(flux, spi, ref_freq, frequency):
    nsrc = 10
    nchan = 16

    I = flux(nsrc)  # noqa
    freq = frequency(nchan)
    ref_freq = ref_freq(nsrc)
    spi = spi(nsrc)

    model = spectral_model(I, spi, ref_freq, freq)
    np_model = numpy_spectral_model(I, spi, ref_freq, freq)
    assert_array_almost_equal(model, np_model)


def test_spectral_model_multiple_spi(flux, ref_freq, frequency):
    nsrc = 10
    nchan = 16
    nspi = 4

    I = flux(nsrc)  # noqa
    spi = 0.7 + np.random.random((nsrc, nspi)) * 0.2
    ref_freq = ref_freq(nsrc)/1e9
    freq = frequency(nchan)/1e9

    model = spectral_model(I, spi, ref_freq, freq)
    np_model = numpy_spectral_model(I, spi, ref_freq, freq)
    assert_array_almost_equal(model, np_model)
