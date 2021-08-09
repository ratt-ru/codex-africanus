import numpy as np
from numpy.testing import assert_allclose
import pytest

from africanus.rime.phase import phase_delay
from africanus.model.spectral import spectral_model
from africanus.model.coherency import convert

from africanus.rime.monolothic.rime import rime_factory
from africanus.rime.monolothic.parser import parse_rime


@pytest.mark.parametrize("rime_spec", [
    # "G_{p}[E_{stpf}L_{tpf}K_{stpqf}B_{spq}L_{tqf}E_{q}]G_{q}",
    # "Gp[EpLpKpqBpqLqEq]sGq",
    "[Gp, (Ep, Lp, Kpq, Bpq, Lq, Eq), Gq]: [I, Q, U, V] -> [XX, XY, YX, YY]",
    # "[Gp x (Ep x Lp x Kpq x Bpq x Lq x Eq) x Gq] -> [XX, XY, YX, YY]",
])
def test_rime_parser(rime_spec):
    # custom_mapping = {"Kpq": MyCustomPhaseTerm}
    print(parse_rime(rime_spec))


def test_monolithic_rime():
    nsrc = 10
    nrow = 5
    nspi = 2
    nchan = 4

    lm = np.random.random(size=(nsrc, 2))*1e-5
    uvw = np.random.random(size=(nrow, 3))
    chan_freq = np.linspace(.856e9, 2*.859e9, nchan)
    stokes = np.random.random(size=(nsrc, 4))
    spi = np.random.random(size=(nsrc, nspi, stokes.shape[1]))
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
