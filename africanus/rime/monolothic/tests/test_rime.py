import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from africanus.rime.monolothic.rime import rime_factory
from africanus.rime.monolothic.phase import PhaseTerm
from africanus.rime.monolothic.parser import parse_rime


@pytest.mark.parametrize("rime_spec", [
    # "G_{p}[E_{stpf}L_{tpf}K_{stpqf}B_{spq}L_{tqf}E_{q}]G_{q}",
    # "Gp[EpLpKpqBpqLqEq]sGq", 
    "[Gp, (Ep, Lp, Kpq, Bpq, Lq, Eq), Gq]: [I, Q, U, V] -> [XX, XY, YX, YY]",
    # "[Gp x (Ep x Lp x Kpq x Bpq x Lq x Eq) x Gq] -> [XX, XY, YX, YY]",
])
def test_rime_parser(rime_spec):
    #custom_mapping = {"Kpq": MyCustomPhaseTerm}
    #print(parse_rime(rime_spec))
    pass


def test_monolithic_rime():
    # lm = np.random.random(size=(10, 2))*1e-5
    # uvw = np.random.random(size=(5, 3))
    #lm = np.zeros((10, 2))
    nsrc = 1
    nrow = 5
    nspi = 2
    nchan = 4
    
    lm = np.zeros((nsrc, 2))
    uvw = np.ones((nrow, 3))
    chan_freq = np.linspace(.856e9, 2*.859e9, nchan)
    stokes = np.array([[1, 2, 3, 4]])
    spi = np.zeros((nsrc, nspi, stokes.shape[1]))
    ref_freq = np.ones(nsrc)

    rime = rime_factory()
    out = rime(lm=lm, uvw=uvw, chan_freq=chan_freq, stokes=stokes,
               spi=spi, ref_freq=ref_freq,
               convention="casa", spi_base="standard")
    expected, out = np.broadcast_arrays([[[3+0j, 3+4j, 3-4j, -1+0j]]], out)
    assert_almost_equal(expected, out)

    out = rime(lm=lm, uvw=uvw, chan_freq=chan_freq, stokes=stokes,
               spi=spi, ref_freq=ref_freq, convention="fourier")
    expected, out = np.broadcast_arrays([[[3+0j, 3+4j, 3-4j, -1+0j]]], out)
    assert_almost_equal(expected, out)

    out = rime(lm=lm, uvw=uvw, chan_freq=chan_freq, stokes=stokes,
               spi=spi, ref_freq=ref_freq)
    expected, out = np.broadcast_arrays([[[3+0j, 3+4j, 3-4j, -1+0j]]], out)
    assert_almost_equal(expected, out)

    with open("rime_asm.txt", "w") as f:
        print(list(rime.impl.inspect_asm().values())[0], file=f)
