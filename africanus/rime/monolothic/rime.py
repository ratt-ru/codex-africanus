from numba import generated_jit, types
import numpy as np

from africanus.rime.monolothic.intrinsics import term_factory
from africanus.rime.monolothic.terms import Term


class rime_factory:
    def __init__(self):
        from africanus.rime.monolothic.phase import PhaseTerm
        from africanus.rime.monolothic.brightness import BrightnessTerm
        terms = [PhaseTerm(), BrightnessTerm(), PhaseTerm(), BrightnessTerm()]

        for t in terms:
            if not isinstance(t, Term):
                raise TypeError(f"{t} is not of type {Term}")

        if not any(isinstance(t, PhaseTerm) for t in terms):
            raise ValueError("RIME must at least contain a Phase Term")

        if not any(isinstance(t, BrightnessTerm) for t in terms):
            raise ValueError("RIME must at least contain a Brightness Term")

        rime_args = set(a for t in terms for a in t.term_args)
        rime_args = list(sorted(rime_args))

        term_kwarg_set = set(a for t in terms
                       for a in getattr(t, "term_kwargs", ()))
        term_kwargs = list(sorted(term_kwarg_set))

        print("OPTIONAL ARGS", term_kwargs)
        arg_map = {a: i for i, a in enumerate(rime_args)}
        term_arg_inds = tuple(tuple(arg_map[a]
                              for a in t.term_args)
                              for t in terms)

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

            if len(args[0]) < len(rime_args):
                raise ValueError("Insufficient required arguments supplied to RIME")

            if (len(args[0]) - len(rime_args)) % 2 != 0:
                raise ValueError("Invalid keyword argument setup")

            for k in args[0][(len(rime_args))::2]:
                if not isinstance(k, types.StringLiteral):
                    raise ValueError(f"{k} must be a String Literal")

                if k.literal_value not in term_kwarg_set:
                    raise ValueError(f"{k.literal_value} is not a valid kwarg")

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
        self.args = rime_args
        self.arg_map = arg_map
        self.term_kwarg_set = term_kwarg_set
        self.impl = rime

    def __call__(self, **kwargs):
        # Call the implementation
        try:
            args = tuple(kwargs.pop(a) for a in self.args)
        except KeyError as e:
            raise ValueError(f"{str(e)} is a required argument")

        kw = tuple(e for (k, v) in kwargs.items()
                   if k in self.term_kwarg_set
                   for e in (types.literal(k), v))

        return self.impl(*args, *kw)