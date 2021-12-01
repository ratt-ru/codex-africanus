import numba
from numba import generated_jit, types
import numpy as np

from africanus.rime.fused.arguments import ArgumentDependencies
from africanus.rime.fused.intrinsics import IntrinsicFactory
from africanus.rime.fused.specification import RimeSpecification


def rime_impl_factory(terms, transformers, ncorr):
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
        argdeps = ArgumentDependencies(names, terms, transformers)
        factory = IntrinsicFactory(argdeps)
        pack_arguments = factory.pack_argument_fn()
        term_state = factory.term_state_fn()
        term_sampler = factory.term_sampler_fn()

        try:
            lm_i = argdeps.output_names.index("lm")
            uvw_i = argdeps.output_names.index("uvw")
            chan_freq_i = argdeps.output_names.index("chan_freq")
        except ValueError as e:
            raise ValueError(f"{str(e)} is required")

        def impl(names, *inargs):
            args = pack_arguments(inargs)
            state = term_state(args)

            nsrc, _ = args[lm_i].shape
            nrow, _ = args[uvw_i].shape
            nchan, = args[chan_freq_i].shape

            vis = np.zeros((nrow, nchan, ncorr), np.complex128)
            # Kahan summation compensation
            compensation = np.zeros_like(vis)

            for s in range(nsrc):
                for r in range(nrow):
                    t = state.time_index[r]
                    a1 = state.antenna1[r]
                    a2 = state.antenna2[r]
                    f1 = state.feed1[r]  # noqa
                    f2 = state.feed2[r]  # noqa

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

    return rime


class RimeFactory:
    REQUIRED_ARGS = ArgumentDependencies.REQUIRED_ARGS
    REQUIRED_ARGS_LITERAL = tuple(types.literal(n) for n in REQUIRED_ARGS)
    REQUIRED_DASK_SCHEMA = {n: ("row",) for n in REQUIRED_ARGS}
    DEFAULT_SPEC = "(Kpq, Bpq): [I, Q, U, V] -> [XX, XY, YX, YY]"

    def __reduce__(self):
        return (RimeFactory, (self.rime_spec,))

    def __hash__(self):
        return hash(self.rime_spec)

    def __eq__(self, rhs):
        return (isinstance(rhs, RimeFactory) and
                self.rime_spec == rhs.rime_spec)

    def __init__(self, rime_spec=DEFAULT_SPEC):
        if isinstance(rime_spec, RimeSpecification):
            pass
        elif isinstance(rime_spec, (list, tuple)):
            rime_spec = RimeSpecification(*rime_spec)
        elif isinstance(rime_spec, str):
            rime_spec = RimeSpecification(rime_spec)

        self.rime_spec = rime_spec
        self.impl = rime_impl_factory(
            rime_spec.terms,
            rime_spec.transformers,
            len(rime_spec.corrs))

    def dask_blockwise_args(self, **kwargs):
        """ Get the dask schema """
        argdeps = ArgumentDependencies(tuple(kwargs.keys()),
                                       self.rime_spec.terms,
                                       self.rime_spec.transformers)
        dask_schema = {a: ("row",) for a in argdeps.REQUIRED_ARGS}
        # Holds kwargs + any dummy outputs from transformations
        dummy_kw = kwargs.copy()

        for transformer in argdeps.can_create.values():
            kw = {a: dummy_kw[a] for a in transformer.ARGS}
            kw.update((a, kwargs.get(a, d)) for a, d
                      in transformer.KWARGS.items())
            inputs, outputs = transformer.dask_schema(**kw)
            dask_schema.update(inputs)
            dummy_kw.update(outputs)

        for term in self.rime_spec.terms:
            kw = {a: dummy_kw[a] for a in term.ALL_ARGS if a in dummy_kw}
            dask_schema.update(term.dask_schema(**kw))

        names = list(sorted(argdeps.valid_inputs | set(kwargs.keys())))
        blockwise_args = [e for n in names if n in kwargs
                          for e in (kwargs[n], dask_schema.get(n, None))]

        return names, blockwise_args

    def __call__(self, time, antenna1, antenna2, feed1, feed2, **kwargs):
        keys = (self.REQUIRED_ARGS_LITERAL +
                tuple(map(types.literal, kwargs.keys())))

        return self.impl(keys, time,
                         antenna1, antenna2,
                         feed1, feed2,
                         *kwargs.values())
