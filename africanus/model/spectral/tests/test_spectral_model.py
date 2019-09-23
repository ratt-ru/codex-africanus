# -*- coding: utf-8 -*-


import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from africanus.model.spectral.spec_model import (spectral_model,
                                                 numpy_spectral_model)


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


@pytest.mark.parametrize("base", [0, 1, 2, "std", "log", "log10",
                                  ["log", "std", "std", "std"]])
@pytest.mark.parametrize("npol", [0, 1, 2, 4])
def test_spectral_model_multiple_spi(flux, ref_freq, frequency, base, npol):
    nsrc = 10
    nchan = 16
    nspi = 6

    # Clip to number of polarisations
    if isinstance(base, list):
        base = base[0] if npol == 0 else base[:npol]

    if npol > 0:
        flux_slice = (slice(None), None)
        flux_shape = (nsrc, npol)
        spi_shape = (nsrc, nspi, npol)
    else:
        flux_slice = (slice(None),)
        flux_shape = (nsrc,)
        spi_shape = (nsrc, nspi)

    I = flux(nsrc)  # noqa
    stokes = np.broadcast_to(I[flux_slice], flux_shape)  # noqa
    spi = 0.7 + np.random.random(spi_shape) * 0.2
    ref_freq = ref_freq(nsrc)
    freq = frequency(nchan)

    model = spectral_model(stokes, spi, ref_freq, freq, base=base)
    np_model = numpy_spectral_model(stokes, spi, ref_freq, freq, base=base)

    assert_array_almost_equal(model, np_model)
    assert model.flags.c_contiguous is True


@pytest.mark.parametrize("base", [0, 1, 2, "std", "log", "log10",
                                  ["log", "std", "std", "std"]])
@pytest.mark.parametrize("npol", [0, 1, 2, 4])
def test_dask_spectral_model(flux, ref_freq, frequency, base, npol):
    da = pytest.importorskip("dask.array")
    from africanus.model.spectral.spec_model import (
                        spectral_model as np_spectral_model)
    from africanus.model.spectral.dask import spectral_model

    sc = (5, 5)
    fc = (8, 8)
    spc = (6,)

    nsrc = sum(sc)
    nchan = sum(fc)
    nspi = sum(spc)

    # Clip to number of polarisations
    if isinstance(base, list):
        base = base[0] if npol == 0 else base[:npol]

    if npol > 0:
        flux_slice = (slice(None), None)
        flux_shape = (nsrc, npol)
        spi_shape = (nsrc, nspi, npol)
        cc = (npol,)
    else:
        flux_slice = (slice(None),)
        flux_shape = (nsrc,)
        spi_shape = (nsrc, nspi)
        cc = ()

    I = flux(nsrc)  # noqa
    stokes = np.broadcast_to(I[flux_slice], flux_shape)  # noqa
    spi = 0.7 + np.random.random(spi_shape) * 0.2
    ref_freq = ref_freq(nsrc)
    freq = frequency(nchan)

    da_stokes = da.from_array(stokes, chunks=(sc,) + cc)
    da_spi = da.from_array(spi, chunks=(sc, spc) + cc)
    da_ref_freq = da.from_array(ref_freq, chunks=sc)
    da_freq = da.from_array(freq, chunks=fc)

    da_model = spectral_model(da_stokes, da_spi,
                              da_ref_freq, da_freq, base=base)

    np_model = np_spectral_model(stokes, spi, ref_freq, freq, base=base)
    assert_array_almost_equal(da_model, np_model)
