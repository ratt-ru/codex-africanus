from numba import njit, generated_jit
from numba.extending import overload, register_jitable, intrinsic
from numba.experimental import jitclass
import numba as nb
import numpy as np


class Term:
    pass

class Brightness(Term):
    def __init__(self):
        self.args = ["stokes", "chan_freq"]
        self.output = ["brightness"]

    def implementation(self):
        pass

class Phase(Term):
    def __init__(self):
        self.args = ["lm",  "uvw", "chan_freq"]
        self.output = ["phase"]

    def setup(self, lm, uvw, chan_freq):
        dtype = np.result_type(np.dtype(lm.dtype.name), np.dtype(uvw.dtype.name))
        one = lm.dtype(1.0)
        zero = lm.dtype(0.0)

        @register_jitable
        def setup_phase(lm, uvw, chan_freq):
            nsrc = lm.shape[0]
            nrow = uvw.shape[0]
            real_phase = np.empty((nsrc, nrow), dtype=dtype)

            for source in range(nsrc):
                l = lm[source, 0]
                m = lm[source, 1]
                n = one - l**2 - m**2
                n = np.sqrt(n) if n >= zero else zero

                for row in range(nrow):
                    u = uvw[row, 0]
                    v = uvw[row, 1]
                    w = uvw[row, 2]

                    real_phase[source, row] = l*u + m*v + (n - one)*w

            return real_phase

        return setup_phase


    def implementation(self, lm, uvw, chan_freq):
        setup_fn = self.setup(nb.typeof(lm), nb.typeof(uvw), nb.typeof(chan_freq))

        @generated_jit(nopython=True, nogil=True, cache=True)
        def phase(lm, uvw, chan_freq):
            def impl(lm, uvw, chan_freq):
                real_phase = setup_fn(lm, uvw, chan_freq)

                return real_phase

            return impl

        return phase



def phase(lm, uvw, chan_freq):
    pass


class rime_factory:
    def __init__(self):
        terms = [Phase(), Brightness()]
        args = list(sorted(set(a for t in terms for a in t.args)))
        arg_map = {a: i for i, a in enumerate(args)}
        term_arg_inds = tuple(tuple(arg_map[a] for a in t.args) for t in terms)

        print(term_arg_inds)

        # arg_map = nb.typed.Dict.empty(nb.types.unicode_type, nb.uint32)

        # for i, a in enumerate(args):
        #     arg_map[a] = np.uint32(i)

        @njit(nogil=True, cache=True)
        def impl(*args):
            pass

        self.terms = terms
        self.args = args
        self.arg_map = arg_map
        self.impl = impl


    def __call__(self, **kwargs):
        try:
            args = tuple(kwargs[a] for a in self.args)
        except KeyError as e:
            raise ValueError(f"{e} is a required kwarg")
        else:
            self.impl(*args)

if __name__ == "__main__":
    fn = rime_factory()
    lm = np.random.random(size=(10, 2))
    uvw = np.random.random(size=(5, 3))
    chan_freq = np.linspace(.856e9, 2*.859e9, 4)
    stokes = np.random.random(size=(10, 4))

    fn(lm=lm, uvw=uvw, chan_freq=chan_freq, stokes=stokes)

