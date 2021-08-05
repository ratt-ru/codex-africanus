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
    lm = np.zeros((1, 2))
    uvw = np.ones((5, 3))
    chan_freq = np.linspace(.856e9, 2*.859e9, 4)
    #stokes = np.random.random(size=(10, 4))
    stokes = np.array([[1, 2, 3, 4]])

    rime = rime_factory()
    out = rime(lm=lm, uvw=uvw, chan_freq=chan_freq, stokes=stokes, convention="casa")
    out = rime(lm=lm, uvw=uvw, chan_freq=chan_freq, stokes=stokes, convention="fourier")
    out = rime(lm=lm, uvw=uvw, chan_freq=chan_freq, stokes=stokes)

    expected, out = np.broadcast_arrays([[[3+0j, 3+4j, 3-4j, -1+0j]]], out)
    assert_almost_equal(expected, out)
