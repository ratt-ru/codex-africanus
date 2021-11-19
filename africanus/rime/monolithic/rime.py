import numba
from numba import generated_jit, types
import numpy as np

from africanus.rime.monolithic.intrinsics import IntrinsicFactory
from africanus.rime.monolithic.terms.core import Term


class rime_factory:
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
        def rime(arg_names, *inargs):
            if len(inargs) != 1 or not isinstance(inargs[0], types.BaseTuple):
                raise ValueError(f"{inargs[0]} must be be a Tuple")

            assert len(arg_names) == len(inargs[0])
            assert all(isinstance(n, types.Literal) for n in arg_names)
            assert all(n.literal_type is types.unicode_type for n in arg_names)
            arg_names = tuple(n.literal_value for n in arg_names)

            factory = IntrinsicFactory(arg_names, terms, transformers)
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

            def impl(arg_names, *inargs):
                args = pack_arguments(inargs)
                state = term_state(args)

                nsrc, _ = args[lm_i].shape
                nrow, _ = args[uvw_i].shape
                nchan, = args[chan_freq_i].shape
                _, ncorr = args[stokes_i].shape

                vis = np.zeros((nrow, nchan, ncorr), np.complex128)
                compensation = np.zeros_like(vis)  # Kahan summation compensation

                for s in range(nsrc):
                    # it = enumerate(zip(time, antenna1, antenna2))
                    # for r, (t, a1, a2) in it:
                    for r in range(nrow):
                        for f in range(nchan):
                            X = term_sampler(state, s, r, 0, 0, 0, f)

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

    def __call__(self, **kwargs):
        keys = tuple(types.literal(k) for k in kwargs.keys())
        return self.impl(keys, *kwargs.values())
