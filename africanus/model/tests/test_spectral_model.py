# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from africanus.compatibility import PY2, PY3, string_types
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


@pytest.mark.parametrize("base", [0, 1, 2] + ["std", "log", "log10"])
def test_spectral_model_multiple_spi(flux, ref_freq, frequency, base):
    nsrc = 10
    nchan = 16
    nspi = 4

    I = flux(nsrc)  # noqa
    spi = 0.7 + np.random.random((nsrc, nspi)) * 0.2
    ref_freq = ref_freq(nsrc)
    freq = frequency(nchan)

    # Expect failure for string bases in python 2
    if PY2 and isinstance(base, string_types):
        with pytest.raises(TypeError) as exc_info:
            spectral_model(I, spi, ref_freq, freq, base=base)
            assert 'unsupported in python 2' in str(exc_info.value)

        return

    model = spectral_model(I, spi, ref_freq, freq, base=base)
    np_model = numpy_spectral_model(I, spi, ref_freq, freq, base=base)
    assert_array_almost_equal(model, np_model)
