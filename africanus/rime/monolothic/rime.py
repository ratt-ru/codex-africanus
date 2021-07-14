from numba import generated_jit
from numba.core import compiler, cgutils, errors, types
from numba.extending import intrinsic
from numba.core.typed_passes import type_inference_stage
import numpy as np


def scalar_scalar(lhs, rhs):
    return lhs*rhs

def scalar_diag(lhs, rhs):
        return lhs*rhs[0], lhs*rhs[1]

def scalar_full(lhs, rhs):
        return lhs*rhs[0], lhs*rhs[1], lhs*rhs[2], lhs*rhs[3]

def diag_scalar(lhs, rhs):
    return lhs[0]*rhs, lhs[1]*rhs

def diag_diag(lhs, rhs):
    return lhs[0]*rhs[0], lhs[1]*rhs[1]

def diag_full(lhs, rhs):
    raise NotImplementedError

def full_scalar(lhs, rhs):
    return lhs[0]*rhs, lhs[1]*rhs, lhs[2]*rhs, lhs[3]*rhs

def full_diag(lhs, rhs):
    raise NotImplementedError

def full_full(lhs, rhs):
    return (
        lhs[0]*rhs[0] + lhs[1]*rhs[2],
        lhs[0]*rhs[1] + lhs[1]*rhs[3],
        lhs[2]*rhs[0] + lhs[3]*rhs[2],
        lhs[2]*rhs[1] + lhs[3]*rhs[3])

_jones_typ_map = {
    ("scalar", "scalar"): scalar_scalar,
    ("scalar", "diag"): scalar_diag,
    ("scalar", "full"): scalar_full,
    ("diag", "scalar"): diag_scalar,
    ("diag", "diag"): diag_diag,
    ("diag", "full"): diag_full,
    ("full", "scalar"): full_scalar,
    ("full", "diag"): full_diag,
    ("full", "full"): full_full
}

def classify_arg(arg):
    if isinstance(arg, types.Number):
        return "scalar"
    elif isinstance(arg, types.Tuple):
        if len(arg) == 2:
            return "diag"
        elif len(arg) == 4:
            return "full"

    return None


def term_mul(lhs, rhs):
    lhs_type = classify_arg(lhs)
    rhs_type = classify_arg(rhs)

    try:
        return _jones_typ_map[(lhs_type, rhs_type)]
    except KeyError:
        raise errors.TypingError(f"No known multiplication "
                                 f"function for {lhs} and {rhs}")

def unify_jones_terms(typingctx, lhs, rhs):
    lhs_type = classify_arg(lhs)
    rhs_type = classify_arg(rhs)

    corr_map = {"scalar": 1, "diag": 2, "full": 4}

    try:
        lhs_corrs = corr_map[lhs_type]
        rhs_corrs = corr_map[rhs_type]
    except KeyError:
        raise errors.TypingError(f"{lhs} or {rhs} has no mapping")

    lhs_types = (lhs,) if lhs_corrs == 1 else tuple(lhs)
    rhs_types = (rhs,) if rhs_corrs == 1 else tuple(rhs)

    out_type = typingctx.unify_types(*lhs_types, *rhs_types)
    out_corrs = max(lhs_corrs, rhs_corrs)

    return out_type if out_corrs == 1 else types.Tuple((out_type,)*out_corrs)


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

            err = errors.TypingError(
                f"{sampler} should return:\n"
                f"(1) a single scalar correlation\n"
                f"(2) a tuple containing 2 scalar correlations\n"
                f"(3) a tuple containing 4 scalar correlations\n"
                f"but instead got a {typ}")

            if isinstance(typ, types.BaseTuple):
                if len(typ) not in (2, 4):
                    raise err

                if not all(isinstance(e, types.Number) for e in typ):
                    raise err

                continue

            raise err

        sig = types.float64(state, s, r, t, a1, a2, c)

        def codegen(context, builder, signature, args):
            [state, s, r, t, a1, a2, c] = args

            jones = []

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
                jones.append(data)

            for lhs, rhs in zip(sampler_return_types[:-1], sampler_return_types[1:]):
                out_type = unify_jones_terms(typingctx, lhs, rhs)

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
