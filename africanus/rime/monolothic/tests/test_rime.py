import numpy as np

from africanus.rime.monolothic.rime import rime_factory


def test_monolithic_rime():
    lm = np.random.random(size=(10, 2))*1e-5
    uvw = np.random.random(size=(5, 3))
    chan_freq = np.linspace(.856e9, 2*.859e9, 4)
    stokes = np.random.random(size=(10, 4))

    rime = rime_factory()
    out = rime(lm=lm, uvw=uvw, chan_freq=chan_freq, stokes=stokes)  # noqa
