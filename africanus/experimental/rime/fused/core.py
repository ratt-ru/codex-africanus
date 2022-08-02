from collections.abc import Mapping
from collections import defaultdict

import numba
from numba import generated_jit, types
import numpy as np

from africanus.util.patterns import Multiton
from africanus.experimental.rime.fused.arguments import ArgumentDependencies
from africanus.experimental.rime.fused.intrinsics import IntrinsicFactory
from africanus.experimental.rime.fused.specification import RimeSpecification


DATASET_TYPES = []

try:
    from daskms.dataset import Dataset as dmsds
except ImportError:
    pass
else:
    DATASET_TYPES.append(dmsds)

try:
    from xarray import Dataset as xrds
except ImportError:
    pass
else:
    DATASET_TYPES.append(xrds)


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
        out_names, pack_opts_indices = factory.pack_optionals_and_indices_fn()
        out_names, pack_transformed = factory.pack_transformed_fn(out_names)
        term_state = factory.term_state_fn(out_names)
        term_sampler = factory.term_sampler_fn()

        try:
            lm_i = out_names.index("lm")
            uvw_i = out_names.index("uvw")
            chan_freq_i = out_names.index("chan_freq")
        except ValueError as e:
            raise ValueError(f"{str(e)} is required")

        def impl(names, *inargs):
            args_opt_idx = pack_opts_indices(inargs)
            args = pack_transformed(args_opt_idx)
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
                    f1 = state.feed1[r]
                    f2 = state.feed2[r]

                    for ch in range(nchan):
                        X = term_sampler(state, s, r, t, f1, f2, a1, a2, ch)

                        for c, value in enumerate(numba.literal_unroll(X)):
                            # Kahan summation
                            y = value - compensation[r, ch, c]
                            current = vis[r, ch, c]
                            x = current + y
                            compensation[r, ch, c] = (x - current) - y
                            vis[r, ch, c] = x

            return vis

        return impl

    return rime


class RimeFactory(metaclass=Multiton):
    REQUIRED_ARGS = ArgumentDependencies.REQUIRED_ARGS
    REQUIRED_ARGS_LITERAL = tuple(types.literal(n) for n in REQUIRED_ARGS)
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
        argdeps = ArgumentDependencies(
            tuple(kwargs.keys()),
            self.rime_spec.terms,
            self.rime_spec.transformers)
        # Holds kwargs + any dummy outputs from transformations
        dummy_kw = kwargs.copy()

        dask_schema = defaultdict(list)
        for a in argdeps.REQUIRED_ARGS:
            dask_schema[a].append(("internal", ("row",)))

        POISON = object()

        for transformer in argdeps.can_create.values():
            kw = {}

            for a in transformer.ARGS:
                v = dummy_kw.get(a, None if a in argdeps.KEY_ARGS else POISON)
                kw[a] = v

            for a, d in transformer.KWARGS.items():
                kw[a] = dummy_kw.get(a, d)

            inputs, outputs = transformer.dask_schema(**kw)

            for k, schema in inputs.items():
                dask_schema[k].append((transformer, schema))

            dummy_kw.update(outputs)

        for term in self.rime_spec.terms:
            kw = {a: dummy_kw[a] for a in term.ALL_ARGS if a in dummy_kw}

            for k, v in term.dask_schema(**kw).items():
                dask_schema[k].append((term, v))

        merged_schema = {}

        for a, candidates in dask_schema.items():
            dims = set(pair[1] for pair in candidates)
            if len(dims) != 1:
                raise ValueError(
                    f"Multiple candidates provided conflicting "
                    f"dimension definitions for {a}: {candidates}.")

            merged_schema[a] = dims.pop()

        names = list(sorted(argdeps.valid_inputs & set(kwargs.keys())))
        blockwise_args = [e for n in names
                          for e in (kwargs[n], merged_schema.get(n, None))]

        assert 2 * len(names) == len(blockwise_args)
        return names, blockwise_args

    def __call__(self, time, antenna1, antenna2, feed1, feed2, **kwargs):
        keys = (self.REQUIRED_ARGS_LITERAL +
                tuple(map(types.literal, kwargs.keys())))

        return self.impl(keys, time,
                         antenna1, antenna2,
                         feed1, feed2,
                         *kwargs.values())


def consolidate_args(args, kw):
    mapping = {}
    oargs = []

    for element in args:
        if isinstance(element, tuple(DATASET_TYPES)):
            mapping.update((k.lower(), v.data) for k, v in element.items())
        elif isinstance(element, Mapping):
            mapping.update(element)
        else:
            oargs.append(element)

    mapping.update(zip(oargs, RimeFactory.REQUIRED_ARGS))
    mapping.update(kw)

    return mapping


def rime(rime_spec, *args, **kw):
    """
    Evaluates the Radio Interferometer Measurement Equation (RIME), given
    the Specification of the RIME :code:`rime_spec`, as well as the
    inputs to the RIME given in :code:`*args` and :code:`**kwargs`.
    """
    mapping = consolidate_args(args, kw)
    factory = RimeFactory(rime_spec=rime_spec)
    return factory(**mapping)
