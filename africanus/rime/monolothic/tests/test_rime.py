import numpy as np
from numpy.testing import (
    assert_array_almost_equal,
    assert_array_almost_equal_nulp)
import pytest

from africanus.rime.phase import phase_delay
from africanus.model.spectral import spectral_model
from africanus.model.coherency import convert

from africanus.rime.monolothic.rime import rime_factory
from africanus.rime.monolothic.dask import rime as dask_rime


@pytest.mark.parametrize("rime_spec", [
    # "G_{p}[E_{stpf}L_{tpf}K_{stpqf}B_{spq}L_{tqf}E_{q}]G_{q}",
    # "Gp[EpLpKpqBpqLqEq]sGq",
    "[Gp, (Ep, Lp, Kpq, Bpq, Lq, Eq), Gq]: [I, Q, U, V] -> [XX, XY, YX, YY]",
    # "[Gp x (Ep x Lp x Kpq x Bpq x Lq x Eq) x Gq] -> [XX, XY, YX, YY]",
])
def test_rime_parser(rime_spec):
    # custom_mapping = {"Kpq": MyCustomPhaseTerm}
    # print(parse_rime(rime_spec))
    pass


chunks = {
    "source": (2, 2, 2, 3),
    "row": (2, 2, 2, 2),
    "spi": (2,),
    "chan": (2, 2),
    "corr": (4,),
    "lm": (2,),
    "uvw": (3,),
}


@pytest.mark.parametrize("chunks", [chunks])
def test_monolithic_rime(chunks):
    nsrc = sum(chunks["source"])
    nrow = sum(chunks["row"])
    nspi = sum(chunks["spi"])
    nchan = sum(chunks["chan"])
    ncorr = sum(chunks["corr"])

    lm = np.random.random(size=(nsrc, 2))*1e-5
    uvw = np.random.random(size=(nrow, 3))
    chan_freq = np.linspace(.856e9, 2*.859e9, nchan)
    stokes = np.random.random(size=(nsrc, ncorr))
    spi = np.random.random(size=(nsrc, nspi, ncorr))
    ref_freq = np.random.random(size=nsrc)*.856e9

    rime = rime_factory()
    out = rime(lm=lm, uvw=uvw, chan_freq=chan_freq, stokes=stokes,
               spi=spi, ref_freq=ref_freq,
               convention="casa", spi_base="standard")
    P = phase_delay(lm, uvw, chan_freq, convention="casa")
    SM = spectral_model(stokes, spi, ref_freq, chan_freq, base="std")
    B = convert(SM, ["I", "Q", "U", "V"], ["XX", "XY", "YX", "YY"])
    expected = (P[:, :, :, None]*B[:, None, :, :]).sum(axis=0)
    assert_array_almost_equal_nulp(expected, out)

    out = rime(lm=lm, uvw=uvw, chan_freq=chan_freq, stokes=stokes,
               spi=spi, ref_freq=ref_freq, convention="fourier")

    P = phase_delay(lm, uvw, chan_freq, convention="fourier")
    SM = spectral_model(stokes, spi, ref_freq, chan_freq, base="std")
    B = convert(SM, ["I", "Q", "U", "V"], ["XX", "XY", "YX", "YY"])
    expected = (P[:, :, :, None]*B[:, None, :, :]).sum(axis=0)
    assert_array_almost_equal_nulp(expected, out)

    out = rime(lm=lm, uvw=uvw, chan_freq=chan_freq, stokes=stokes,
               spi=spi, ref_freq=ref_freq)
    P = phase_delay(lm, uvw, chan_freq, convention="fourier")
    SM = spectral_model(stokes, spi, ref_freq, chan_freq, base="std")
    B = convert(SM, ["I", "Q", "U", "V"], ["XX", "XY", "YX", "YY"])
    expected = (P[:, :, :, None]*B[:, None, :, :]).sum(axis=0)
    assert_array_almost_equal_nulp(expected, out)


@pytest.mark.parametrize("chunks", [chunks])
def test_monolithic_dask_rime(chunks):
    da = pytest.importorskip("dask.array")

    nsrc = sum(chunks["source"])
    nrow = sum(chunks["row"])
    nspi = sum(chunks["spi"])
    nchan = sum(chunks["chan"])
    ncorr = sum(chunks["corr"])

    lm = np.random.random(size=(nsrc, 2))*1e-5
    uvw = np.random.random(size=(nrow, 3))*1e5
    chan_freq = np.linspace(.856e9, 2*.859e9, nchan)
    stokes = np.random.random(size=(nsrc, ncorr))
    spi = np.random.random(size=(nsrc, nspi, ncorr))
    ref_freq = np.random.random(size=nsrc)*.856e9

    def darray(array, dims):
        return da.from_array(array, tuple(chunks[d] for d in dims))

    dask_lm = darray(lm, ("source", "lm"))
    dask_uvw = darray(uvw, ("row", "uvw"))
    dask_chan_freq = darray(chan_freq, ("chan",))
    dask_stokes = darray(stokes, ("source", "corr"))
    dask_spi = darray(spi, ("source", "spi", "corr"))
    dask_ref_freq = darray(ref_freq, ("source",))

    dask_out = dask_rime(lm=dask_lm, uvw=dask_uvw, stokes=dask_stokes,
                         spi=dask_spi, chan_freq=dask_chan_freq,
                         ref_freq=dask_ref_freq, convention="casa")

    rime = rime_factory()
    out = rime(lm=lm, uvw=uvw, chan_freq=chan_freq, stokes=stokes,
               spi=spi, ref_freq=ref_freq, convention="casa")

    assert_array_almost_equal(dask_out, out, decimal=5)
