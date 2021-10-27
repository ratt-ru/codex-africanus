import numba
from numba import generated_jit, types
import numpy as np

from africanus.rime.monolithic.argpack import pack_arguments
from africanus.rime.monolithic.intrinsics import extend_argpack, term_factory
from africanus.rime.monolithic.terms import Term


class rime_factory:
    def __init__(self, terms=None):
        from africanus.rime.monolithic.phase import PhaseTerm
        from africanus.rime.monolithic.brightness import BrightnessTerm
        terms = terms or [PhaseTerm(), BrightnessTerm()]

        for t in terms:
            if not isinstance(t, Term):
                raise TypeError(f"{t} is not of type {Term}")

        if not any(isinstance(t, PhaseTerm) for t in terms):
            raise ValueError("RIME must at least contain a Phase Term")

        if not any(isinstance(t, BrightnessTerm) for t in terms):
            raise ValueError("RIME must at least contain a Brightness Term")

        expected_args = set(a for t in terms for a in t.ARGS)
        expected_args = list(sorted(expected_args))

        extra_args_set = set(k for t in terms for k in t.KWARGS)
        arg_map = {a: i for i, a in enumerate(expected_args)}

        @generated_jit(nopython=True, nogil=True, cache=False)
        def rime(*inargs):
            global extend_argpack

            if len(inargs) != 1 or not isinstance(inargs[0], types.BaseTuple):
                raise ValueError(f"{inargs[0]} must be be a Tuple")

            arg_pack = pack_arguments(terms, inargs[0])
            new_argpack, extend_argpack_impl = extend_argpack(arg_pack)
            state_factory, pairwise_sample = term_factory(new_argpack, terms)

            try:
                lm_i = new_argpack.index("lm")
                uvw_i = new_argpack.index("uvw")
                chan_freq_i = new_argpack.index("chan_freq")
                stokes_i = new_argpack.index("stokes")
            except KeyError as e:
                raise ValueError(f"'{str(e)}' is a required argument")

            def impl(*inargs):
                args = extend_argpack_impl(inargs)
                state = state_factory(args)  # noqa: F841

                nsrc, _ = args[lm_i].shape
                nrow, _ = args[uvw_i].shape
                nchan, = args[chan_freq_i].shape
                _, ncorr = args[stokes_i].shape

                vis = np.zeros((nrow, nchan, ncorr), np.complex128)

                # it = enumerate(zip(time, antenna1, antenna2))
                # for r, (t, a1, a2) in it:
                for r in range(nrow):
                    for f in range(nchan):
                        X = pairwise_sample(state, nsrc, r, 0, 0, 0, f)

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
            try:
                args = tuple(kwargs[a] for a in t.ARGS)
            except KeyError as e:
                raise ValueError(f"{str(e)} is a required argument")

            kw = {k: kwargs.get(k, v) for k, v in t.KWARGS.items()}

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
