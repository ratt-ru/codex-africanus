from numba import generated_jit
from numba.core import compiler, cgutils, errors, types
from numba.extending import intrinsic
from numba.core.typed_passes import type_inference_stage
import numpy as np


def term_factory(args, terms, term_arg_inds):
    term_arg_types = tuple(tuple(args[j] for j in idx)
                           for idx in term_arg_inds)
    assert len(terms) == len(term_arg_inds)

    term_state_types = [term.term_type(*arg_types)
                        for term, arg_types
                        in zip(terms, term_arg_types)]

    constructors = [term.initialiser(*arg_types)
                    for term, arg_types
                    in zip(terms, term_arg_types)]

    samplers = [term.sampler() for term in terms]

    @intrinsic
    def construct_terms(typginctx, args):
        if not isinstance(args, types.Tuple):
            raise errors.TypingError("args must be a Tuple")

        return_type = types.Tuple(term_state_types)

        sig = return_type(args)

        def codegen(context, builder, signature, args):
            return_type = signature.return_type

            if not isinstance(return_type, types.Tuple):
                raise errors.TypingError(
                    "signature.return_type should be a Tuple")

            llvm_ret_type = context.get_value_type(return_type)
            ret_tuple = cgutils.get_null_value(llvm_ret_type)

            if not len(args) == 1:
                raise errors.TypingError("args must contain a single value")

            # Our single argument is a tuple of arguments, but we
            # need to extract those arguments necessary to construct
            # the term StructRef
            constructor_args = [[builder.extract_value(args[0], j)
                                 for j in idx]
                                for idx in term_arg_inds]

            # Sanity
            assert all(len(ca) == len(at) for ca, at
                       in zip(constructor_args, term_arg_types))

            for ti in range(return_type.count):
                constructor_sig = return_type[ti](*term_arg_types[ti])
                data = context.compile_internal(builder,
                                                constructors[ti],
                                                constructor_sig,
                                                constructor_args[ti])

                ret_tuple = builder.insert_value(ret_tuple, data, ti)

            return ret_tuple

        return sig, codegen

    @intrinsic
    def sample_terms(typingctx, state, s, r, t, a1, a2, c):
        if not isinstance(state, types.Tuple):
            raise errors.TypingError(f"{state} must be a Tuple")

        nterms = len(state)

        if not nterms == len(term_state_types):
            raise errors.TypingError(
                "State length does not equal the number of terms")

        if not all(st == tst for st, tst in zip(state, term_state_types)):
            raise errors.TypingError("State types don't match")

        sampler_ir = list(map(compiler.run_frontend, samplers))
        idx_types = (s, r, t, a1, a2, c)
        ir_args = [(typ,) + idx_types for typ in term_state_types]
        type_infer = [type_inference_stage(typingctx, ir, args, None)
                      for ir, args in zip(sampler_ir, ir_args)]
        sampler_return_types = [ti.return_type for ti in type_infer]

        # Sanity check the sampler return types
        for typ, sampler in zip(sampler_return_types, samplers):
            if isinstance(typ, types.Number):
                continue

            if isinstance(typ, types.BaseTuple):
                if len(typ) not in (1, 2, 4):
                    raise TypeError(f"{sampler} may only return "
                                    f"1, 2 or 4 correlations. "
                                    f"Got {len(typ)}")

                if not all(isinstance(e, types.Number) for e in typ):
                    raise TypeError(f"{sampler} must return a "
                                    f"Tuple of Numbers. Got {typ}")

                continue

            raise TypeError(f"sampler must return a Number "
                            f"or a Tuple of Numbers. "
                            f"Got {typ}")

        sig = types.float64(state, s, r, t, a1, a2, c)

        def codegen(context, builder, signature, args):
            [state, s, r, t, a1, a2, c] = args

            for ti in range(nterms):
                sampling_fn = samplers[ti]

                # Build signature for the sampling function
                ret_type = sampler_return_types[ti]
                sampler_arg_types = (
                    term_state_types[ti],) + signature.args[1:]
                sampler_sig = ret_type(*sampler_arg_types)

                # Build LLVM arguments for the sampling function
                term_state = builder.extract_value(state, ti)
                sampler_args = [term_state] + [s, r, t, a1, a2, c]

                # Call the sampling function
                data = context.compile_internal(builder,  # noqa
                                                sampling_fn,
                                                sampler_sig,
                                                sampler_args)

            return context.get_constant(types.float64, 10.0)

        return sig, codegen

    return construct_terms, sample_terms


class rime_factory:
    def __init__(self):

        # "G_{p}[E_{stp}K_{stp}B_{s}K_{stq}E_{stq}]sG_{q}"
        from africanus.rime.monolothic.phase import PhaseTerm
        from africanus.rime.monolothic.brightness import BrightnessTerm
        terms = [PhaseTerm, BrightnessTerm]
        args = list(sorted(set(a for t in terms for a in t.term_args)))
        arg_map = {a: i for i, a in enumerate(args)}
        term_arg_inds = tuple(tuple(arg_map[a]
                              for a in t.term_args) for t in terms)

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
                    for r in range(nrow):
                        for f in range(nchan):
                            vis[r, f, 0] += sample_terms(term_state,
                                                         s, r, 0, 0, 0, f)

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
