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

    def from_args(self, index, *args):
        args = tuple((n, args[i]) for for i, n in zip(index, self.args))
        assert len(args) == len(self.args)
        phase_type = np.result_type(a.dtype for _, a in args)
        phase_type = nb.optional(nb.typeof(phase_type))

        spec = [(n, nb.typeof(a)) for n, a in args]
        spec.extend(("phase", phase_type))

        @jitclass(spec=spec)
        class foocls:
            def __init__(self, lm, uvw, chan_freq):
                self.lm = lm
                self.uvw = uvw
                self.chan_freq = chan_freq
                self.phase = None

            def setup():
                lm = self.lm
                uvw = self.uvw

                nsrc = lm.shape[0]
                nrow = uvw.shape[0]
                self.phase = real_phase = np.empty((nsrc, nrow), dtype=phase_type)

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


            return foocls
class rime_factory:
    def __init__(self):
        terms = [Phase(), Brightness()]
        args = list(sorted(set(a for t in terms for a in t.args)))
        arg_map = {a: i for i, a in enumerate(args)}
        term_arg_inds = tuple(tuple(arg_map[a] for a in t.args) for t in terms)

        @njit(nogil=True, cache=True)
        def impl(*args):
            for t in nb.literal_unroll(args):


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
    rime = rime_factory()
    lm = np.random.random(size=(10, 2))
    uvw = np.random.random(size=(5, 3))
    chan_freq = np.linspace(.856e9, 2*.859e9, 4)
    stokes = np.random.random(size=(10, 4))

    rime(lm=lm, uvw=uvw, chan_freq=chan_freq, stokes=stokes)

