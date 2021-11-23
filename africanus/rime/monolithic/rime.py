import numba
from numba import generated_jit, types
import numpy as np

from africanus.rime.monolithic.intrinsics import (
    REQUIRED_ARGS, IntrinsicFactory)
from africanus.rime.monolithic.terms.core import Term


class rime_factory:
    REQUIRED_ARGS = REQUIRED_ARGS
    REQUIRED_ARGS_LITERAL = tuple(types.literal(n) for n in REQUIRED_ARGS)
    REQUIRED_DASK_SCHEMA = {n: ("row",) for n in REQUIRED_ARGS}

    def __init__(self, terms=None, transformers=None):
        from africanus.rime.monolithic.terms.phase import PhaseTerm
        from africanus.rime.monolithic.terms.brightness import BrightnessTerm
        from africanus.rime.monolithic.transformers.lm import LMTransformer
        terms = terms or [PhaseTerm(), BrightnessTerm()]
        transformers = transformers or [LMTransformer()]

        for t in terms:
            if not isinstance(t, Term):
                raise TypeError(f"{t} is not of type {Term}")

        if not any(isinstance(t, PhaseTerm) for t in terms):
            raise ValueError("RIME must at least contain a Phase Term")

        if not any(isinstance(t, BrightnessTerm) for t in terms):
            raise ValueError("RIME must at least contain a Brightness Term")

        @generated_jit(nopython=True, nogil=True, cache=True)
        def rime(names, *inargs):
            if len(inargs) != 1 or not isinstance(inargs[0], types.BaseTuple):
                raise TypeError(f"{inargs[0]} must be be a Tuple")

            if not isinstance(names, types.BaseTuple):
                raise TypeError(f"{names} must be a Tuple of strings")

            if len(names) != len(inargs[0]):
                raise ValueError(f"len(names): {len(names)} "
                                 f"!= {len(inargs[0])}")

            if not all(isinstance(n, types.Literal) for n in names):
                raise TypeError(f"{names} must be a Tuple of strings")

            if not all(n.literal_type is types.unicode_type for n in names):
                raise TypeError(f"{names} must be a Tuple of strings")

            # Get literal argument names
            names = tuple(n.literal_value for n in names)

            # Generate intrinsics
            factory = IntrinsicFactory(names, terms, transformers)
            pack_arguments = factory.pack_argument_fn()
            term_state = factory.term_state_fn()
            term_sampler = factory.term_sampler_fn()

            try:
                lm_i = factory.output_names.index("lm")
                uvw_i = factory.output_names.index("uvw")
                chan_freq_i = factory.output_names.index("chan_freq")
                stokes_i = factory.output_names.index("stokes")
            except ValueError as e:
                raise ValueError(f"{str(e)} is required")

            def impl(names, *inargs):
                args = pack_arguments(inargs)
                state = term_state(args)

                nsrc, _ = args[lm_i].shape
                nrow, _ = args[uvw_i].shape
                nchan, = args[chan_freq_i].shape
                _, ncorr = args[stokes_i].shape

                vis = np.zeros((nrow, nchan, ncorr), np.complex128)
                # Kahan summation compensation
                compensation = np.zeros_like(vis)

                for s in range(nsrc):
                    for r in range(nrow):
                        t = state.time_index[r]
                        a1 = state.antenna1[r]
                        a2 = state.antenna2[r]
                        f1 = state.feed1[r]
                        f2 = state.feed2[r]

                        for f in range(nchan):
                            X = term_sampler(state, s, r, t, a1, a2, f)

                            for c, value in enumerate(numba.literal_unroll(X)):
                                # Kahan summation
                                y = value - compensation[r, f, c]
                                current = vis[r, f, c]
                                t = current + y
                                compensation[r, f, c] = (t - current) - y
                                vis[r, f, c] = t

                return vis

            return impl

        self.impl = rime
        self.terms = terms
        self.transformers = transformers

    def __reduce__(self):
        return (rime_factory, (self.terms, self.transformers))

    def dask_blockwise_args(self, **kwargs):
        """ Get the dask schema """
        factory = IntrinsicFactory(
            tuple(kwargs.keys()), self.terms, self.transformers)
        dask_schema = {}

        for term in self.terms:
            kw = {a: kwargs[a] for a in term.ALL_ARGS if a in kwargs}
            dask_schema.update(term.dask_schema(**kw))

        for _, transformer in factory.can_create.items():
            kw = {a: kwargs[a] for a in transformer.ALL_ARGS if a in kwargs}
            dask_schema.update(transformer.dask_schema(**kw))

        names = list(kwargs.keys())
        blockwise_args = []

        for name in names:
            blockwise_args.append(kwargs[name])
            blockwise_args.append(dask_schema.get(name, None))

        return names, blockwise_args

    def __call__(self, time, antenna1, antenna2, feed1, feed2, **kwargs):
        keys = (self.REQUIRED_ARGS_LITERAL +
                tuple(types.literal(k) for k in kwargs.keys()))
        return self.impl(keys, time,
                         antenna1, antenna2,
                         feed1, feed2,
                         *kwargs.values())
