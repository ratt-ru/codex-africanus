import dask.array as da
import numpy as np
from numpy.testing import assert_allclose
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
    "source": (5, 5),
    "row": (5,),
    "spi": (2,),
    "chan": (4,),
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
    ref_freq = np.random.random(size=nsrc)

    rime = rime_factory()
    out = rime(lm=lm, uvw=uvw, chan_freq=chan_freq, stokes=stokes,
               spi=spi, ref_freq=ref_freq,
               convention="casa", spi_base="standard")
    P = phase_delay(lm, uvw, chan_freq, convention="casa")
    SM = spectral_model(stokes, spi, ref_freq, chan_freq, base="std")
    B = convert(SM, ["I", "Q", "U", "V"], ["XX", "XY", "YX", "YY"])
    expected = (P[:, :, :, None]*B[:, None, :, :]).sum(axis=0)
    assert_allclose(expected, out)

    out = rime(lm=lm, uvw=uvw, chan_freq=chan_freq, stokes=stokes,
               spi=spi, ref_freq=ref_freq, convention="fourier")

    P = phase_delay(lm, uvw, chan_freq, convention="fourier")
    SM = spectral_model(stokes, spi, ref_freq, chan_freq, base="std")
    B = convert(SM, ["I", "Q", "U", "V"], ["XX", "XY", "YX", "YY"])
    expected = (P[:, :, :, None]*B[:, None, :, :]).sum(axis=0)
    assert_allclose(expected, out)

    out = rime(lm=lm, uvw=uvw, chan_freq=chan_freq, stokes=stokes,
               spi=spi, ref_freq=ref_freq)
    P = phase_delay(lm, uvw, chan_freq, convention="fourier")
    SM = spectral_model(stokes, spi, ref_freq, chan_freq, base="std")
    B = convert(SM, ["I", "Q", "U", "V"], ["XX", "XY", "YX", "YY"])
    expected = (P[:, :, :, None]*B[:, None, :, :]).sum(axis=0)
    assert_allclose(expected, out)

    with open("rime_asm.txt", "w") as f:
        print(list(rime.impl.inspect_asm().values())[0], file=f)


@pytest.mark.parametrize("chunks", [chunks])
def test_monolithic_dask_rime(chunks):
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
    ref_freq = np.random.random(size=nsrc)

    achunks = tuple(chunks[d] for d in ("source", "lm"))
    dask_lm = da.from_array(lm, chunks=achunks)

    achunks = tuple(chunks[d] for d in ("row", "uvw"))
    dask_uvw = da.from_array(uvw, chunks=achunks)

    achunks = tuple(chunks[d] for d in ("chan",))
    dask_chan_freq = da.from_array(chan_freq, chunks=achunks)

    achunks = tuple(chunks[d] for d in ("source", "corr"))
    dask_stokes = da.from_array(stokes, chunks=achunks)

    achunks = tuple(chunks[d] for d in ("source", "spi", "corr"))
    dask_spi = da.from_array(spi, chunks=achunks)

    achunks = tuple(chunks[d] for d in ("source",))
    dask_ref_freq = da.from_array(ref_freq, chunks=achunks)

    out = dask_rime(lm=dask_lm, uvw=dask_uvw, stokes=dask_stokes,
                    spi=dask_spi, chan_freq=dask_chan_freq,
                    ref_freq=dask_ref_freq, convention="fourier")

    result = out.compute()
    expected_shape = tuple(sum(chunks[d]) for d in ("row", "chan", "corr"))
    assert result.shape == expected_shape
