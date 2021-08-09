import inspect
from numba import generated_jit, types
import numpy as np

from africanus.rime.monolothic.intrinsics import term_factory
from africanus.rime.monolothic.terms import Term, SignatureAdapter


class rime_factory:
    def __init__(self):
        from africanus.rime.monolothic.phase import PhaseTerm
        from africanus.rime.monolothic.brightness import BrightnessTerm
        terms = [PhaseTerm(), BrightnessTerm()]

        for t in terms:
            if not isinstance(t, Term):
                raise TypeError(f"{t} is not of type {Term}")

        if not any(isinstance(t, PhaseTerm) for t in terms):
            raise ValueError("RIME must at least contain a Phase Term")

        if not any(isinstance(t, BrightnessTerm) for t in terms):
            raise ValueError("RIME must at least contain a Brightness Term")

        signatures = [inspect.signature(t.term_type) for t in terms]
        adapted_sigs = list(map(SignatureAdapter, signatures))

        expected_args = set(a for s in adapted_sigs for a in s.args)
        expected_args = list(sorted(expected_args))

        extra_args_set = set(k for s in adapted_sigs for k in s.kwargs)
        arg_map = {a: i for i, a in enumerate(expected_args)}

        try:
            lm_i = arg_map["lm"]
            uvw_i = arg_map["uvw"]
            chan_freq_i = arg_map["chan_freq"]
            stokes_i = arg_map["stokes"]
        except KeyError as e:
            raise ValueError(f"'{str(e)}' is a required argument")

        @generated_jit(nopython=True, nogil=True)
        def rime(*args):
            if len(args) != 1 or not isinstance(args[0], types.BaseTuple):
                raise ValueError(f"{args[0]} must be be a Tuple")

            n = len(expected_args)
            starargs = args[0]
            kwargs = starargs[n:]
            starargs = starargs[:n]

            if len(starargs) < n:
                raise ValueError("Insufficient required arguments "
                                 "supplied to RIME")

            # Extract kwarg (string, type) pairs after
            # the expected arguments
            if len(kwargs) % 2 != 0:
                raise ValueError(f"Invalid keyword argument pairs "
                                 f"after required arguments "
                                 f"{kwargs}")

            tstarargs = {k: (vt, i) for i, (k, vt)
                         in enumerate(zip(expected_args, starargs))}
            tkwargs = {}
            it = enumerate(zip(kwargs[::2], kwargs[1::2]))

            for i, (k, vt) in it:
                if not isinstance(k, types.StringLiteral):
                    raise ValueError(f"{k} must be a String Literal")

                k = k.literal_value

                if k not in extra_args_set:
                    raise ValueError(f"{k} is not a "
                                     f"recognised keyword argument "
                                     f"to any term in this RIME")

                tkwargs[k] = (vt, 2*i + 1 + n)

            state_factory, sample_terms = term_factory(
                tstarargs, tkwargs, terms)

            def impl(*args):
                term_state = state_factory(args)  # noqa: F841

                nsrc, _ = args[lm_i].shape
                nrow, _ = args[uvw_i].shape
                nchan, = args[chan_freq_i].shape
                _, ncorr = args[stokes_i].shape

                vis = np.zeros((nrow, nchan, ncorr), np.complex128)

                for s in range(nsrc):
                    # it = enumerate(zip(time, antenna1, antenna2))
                    # for r, (t, a1, a2) in it:
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
        self.args = expected_args
        self.arg_map = arg_map
        self.term_kwarg_set = extra_args_set
        self.impl = rime

    def __call__(self, **kwargs):
        # Call the implementation
        try:
            args = tuple(kwargs.pop(a) for a in self.args)
        except KeyError as e:
            raise ValueError(f"{str(e)} is a required argument")

        # Pack any kwargs into a
        # (literal(key1), value1, ... literal(keyn), valuen)
        # sequence after the required arguments
        kw = tuple(e for (k, v) in kwargs.items()
                   if k in self.term_kwarg_set
                   for e in (types.literal(k), v))

        return self.impl(*args, *kw)
