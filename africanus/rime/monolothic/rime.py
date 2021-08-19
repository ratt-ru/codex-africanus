import inspect

import numba
from numba import generated_jit, types
import numpy as np

from africanus.rime.monolothic.intrinsics import term_factory, tuple_adder
from africanus.rime.monolothic.terms import Term, SignatureAdapter

PAIRWISE_BLOCKSIZE = 128


@numba.njit(nopython=True, nogil=True)
def pairwise_sample(sample_terms, state, ss, se, r, t, a1, a2, c):
    """
    This code based on https://github.com/numpy/numpy/pull/3685
    """
    nsrc = se - ss

    if nsrc < 8:
        X = sample_terms(state, 0, r, t, a1, a2, c)

        for s in range(1, nsrc):
            Y = sample_terms(state, s, r, t, a1, a2, c)
            X = tuple_adder(X, Y)

        return X
    elif nsrc <= PAIRWISE_BLOCKSIZE:
        X0 = sample_terms(state, 0, r, t, a1, a2, c)
        X1 = sample_terms(state, 1, r, t, a1, a2, c)
        X2 = sample_terms(state, 2, r, t, a1, a2, c)
        X3 = sample_terms(state, 3, r, t, a1, a2, c)
        X4 = sample_terms(state, 4, r, t, a1, a2, c)
        X5 = sample_terms(state, 5, r, t, a1, a2, c)
        X6 = sample_terms(state, 6, r, t, a1, a2, c)
        X7 = sample_terms(state, 7, r, t, a1, a2, c)

        for s in range(8, nsrc - (nsrc % 8), 8):
            Y0 = sample_terms(state, s + 0, r, t, a1, a2, c)
            Y1 = sample_terms(state, s + 1, r, t, a1, a2, c)
            Y2 = sample_terms(state, s + 2, r, t, a1, a2, c)
            Y3 = sample_terms(state, s + 3, r, t, a1, a2, c)
            Y4 = sample_terms(state, s + 4, r, t, a1, a2, c)
            Y5 = sample_terms(state, s + 5, r, t, a1, a2, c)
            Y6 = sample_terms(state, s + 6, r, t, a1, a2, c)
            Y7 = sample_terms(state, s + 7, r, t, a1, a2, c)

            X0 = tuple_adder(X0, Y0)
            X1 = tuple_adder(X1, Y1)
            X2 = tuple_adder(X2, Y2)
            X3 = tuple_adder(X3, Y3)
            X4 = tuple_adder(X4, Y4)
            X5 = tuple_adder(X5, Y5)
            X6 = tuple_adder(X6, Y6)
            X7 = tuple_adder(X7, Y7)

            Z1 = tuple_adder(tuple_adder(X0, X1), tuple_adder(X2, X3))
            Z2 = tuple_adder(tuple_adder(X4, X5), tuple_adder(X6, X7))
            X = tuple_adder(Z1, Z2)

        while s < nsrc:
            Y = sample_terms(state, s, r, t, a1, a2, c)
            X = tuple_adder(X, Y)
            s += 1

        return X
    else:
        ns2 = (nsrc / 2) - (nsrc % 8)
        X = pairwise_sample(sample_terms, state, ss, ns2,
                            r, t, a1, a2, c)
        Y = pairwise_sample(sample_terms, state, ss + ns2, se - ns2,
                            r, t, a1, a2, c)
        return tuple_adder(X, Y)


class rime_factory:
    def __init__(self, terms=None):
        from africanus.rime.monolothic.phase import PhaseTerm
        from africanus.rime.monolothic.brightness import BrightnessTerm
        terms = terms or [PhaseTerm(), BrightnessTerm()]

        for t in terms:
            if not isinstance(t, Term):
                raise TypeError(f"{t} is not of type {Term}")

        if not any(isinstance(t, PhaseTerm) for t in terms):
            raise ValueError("RIME must at least contain a Phase Term")

        if not any(isinstance(t, BrightnessTerm) for t in terms):
            raise ValueError("RIME must at least contain a Brightness Term")

        signatures = [inspect.signature(t.term_type) for t in terms]

        # Check matching signatures
        for s, t in zip(signatures, terms):
            assert inspect.signature(t.dask_schema) == s
            assert inspect.signature(t.initialiser) == s

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

        # TODO(sjperkins)
        # Reintroduce cache=True when https://github.com/numba/numba/issues/6713
        # is fixed
        @generated_jit(nopython=True, nogil=True, cache=False)
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

                # it = enumerate(zip(time, antenna1, antenna2))
                # for r, (t, a1, a2) in it:
                for r in range(nrow):
                    for f in range(nchan):
                        X = pairwise_sample(sample_terms, term_state, 0, nsrc,
                                            r, 0, 0, 0, f)

                        for c, value in enumerate(numba.literal_unroll(X)):
                            vis[r, f, c] = value

                return vis

            return impl

        self.terms = terms
        self.args = expected_args
        self.arg_map = arg_map
        self.term_kwarg_set = extra_args_set
        self.impl = rime

    def __reduce__(self):
        return (rime_factory, (self.terms,))

    def dask_blockwise_args(self, **kwargs):
        """ Get the dask schema """
        import dask.array as da

        schema = {}

        for t in self.terms:
            sig = SignatureAdapter(inspect.signature(t.dask_schema))

            try:
                args = tuple(kwargs[a] for a in sig.args)
            except KeyError as e:
                raise ValueError(f"{str(e)} is a required argument")

            kw = {k: kwargs.get(k, v) for k, v in sig.kwargs.items()}

            schema.update(t.dask_schema(*args, **kw))

        blockwise_args = []
        names = []

        for a in self.args:
            try:
                arg = kwargs.pop(a)
                arg_schema = schema[a]
            except KeyError as e:
                raise ValueError(f"{str(e)} is a required argument")

            # Do some sanity checks
            if isinstance(arg, da.Array):
                if (not isinstance(arg_schema, tuple) or
                        not len(arg_schema) == arg.ndim):

                    raise ValueError(f"{arg} is a dask array but "
                                     f"associated schema {arg_schema} "
                                     f"doesn't match the number of "
                                     f"dimensions {arg.ndim}")
            else:
                if arg_schema is not None:
                    raise ValueError(f"{arg} is not a dask array but "
                                     f"associated schema {arg_schema} "
                                     f"is not None")

            blockwise_args.append(arg)
            blockwise_args.append(arg_schema)

            names.append(a)

        for k, v in kwargs.items():
            try:
                blockwise_args.append(v)
                blockwise_args.append(schema.get(k, None))
            except KeyError:
                raise ValueError(f"Something went wrong "
                                 f"trying extract kwarg {k}")

            names.append(k)

        return names, blockwise_args

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
