import numpy as np
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
    lm = np.random.random(size=(10, 2))*1e-5
    uvw = np.random.random(size=(5, 3))
    chan_freq = np.linspace(.856e9, 2*.859e9, 4)
    stokes = np.random.random(size=(10, 4))
    stokes[:, 2:] = 0

    rime = rime_factory()
    out = rime(lm=lm, uvw=uvw, chan_freq=chan_freq, stokes=stokes, convention="casa")
    # print(out)
    #print(list(rime.impl.inspect_asm().values())[0])
