from numba import generated_jit
import numpy as np

from africanus.rime.monolothic.intrinsics import term_factory

class rime_factory:
    def __init__(self):
        from africanus.rime.monolothic.phase import PhaseTerm
        from africanus.rime.monolothic.brightness import BrightnessTerm
        terms = [PhaseTerm, BrightnessTerm, PhaseTerm, BrightnessTerm]
        args = list(sorted(set(a for t in terms for a in t.term_args)))
        arg_map = {a: i for i, a in enumerate(args)}
        term_arg_inds = tuple(tuple(arg_map[a]
                              for a in t.term_args) for t in terms)

        if PhaseTerm not in terms:
            raise ValueError("RIME must at least contain a Phase Term")

        if BrightnessTerm not in terms:
            raise ValueError("RIME must at least contain a Brightness Term")

        try:
            lm_i = arg_map["lm"]
            uvw_i = arg_map["uvw"]
            chan_freq_i = arg_map["chan_freq"]
            stokes_i = arg_map["stokes"]
        except KeyError as e:
            raise ValueError(f"'{str(e)}' is a required argument")

        @generated_jit(nopython=True, nogil=True)
        def rime(*args):
            assert len(args) == 1
            state_factory, sample_terms = term_factory(
                args[0], terms, term_arg_inds)

            def impl(*args):
                term_state = state_factory(args)  # noqa: F841

                nsrc, _ = args[lm_i].shape
                nrow, _ = args[uvw_i].shape
                nchan, = args[chan_freq_i].shape
                _, ncorr = args[stokes_i].shape

                vis = np.zeros((nrow, nchan, ncorr), np.complex128)

                for s in range(nsrc):
                    # for  r, (t, a1, a2) in enumerate(zip(time, antenna1, antenna2))
                    for r in range(nrow):
                        for f in range(nchan):
                            X = sample_terms(term_state, s, r, 0, 0, 0, f)

                            vis[r, f, 0] += X[0]
                            vis[r, f, 1] += X[1]
                            vis[r, f, 2] += X[2]
                            vis[r, f, 3] += X[3]

                return vis

            return impl

        self.terms = terms
        self.args = args
        self.arg_map = arg_map
        self.impl = rime

    def __call__(self, **kwargs):
        try:
            args = tuple(kwargs[a] for a in self.args)
        except KeyError as e:
            raise ValueError(f"{e} is a required kwarg")
        else:
            return self.impl(*args)
