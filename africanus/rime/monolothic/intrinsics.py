import inspect

from numba.core import compiler, cgutils, errors, types
from numba.extending import intrinsic, register_jitable
from numba.core.typed_passes import type_inference_stage

from africanus.rime.monolothic.terms import SignatureAdapter, TermStructRef

PAIRWISE_BLOCKSIZE = 128


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
    return (
        lhs[0]*rhs[0],
        lhs[0]*rhs[1],
        lhs[1]*rhs[2],
        lhs[1]*rhs[3])


def full_scalar(lhs, rhs):
    return lhs[0]*rhs, lhs[1]*rhs, lhs[2]*rhs, lhs[3]*rhs


def full_diag(lhs, rhs):
    return (
        lhs[0]*rhs[0],
        lhs[1]*rhs[1],
        lhs[2]*rhs[0],
        lhs[3]*rhs[1])


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
    """
    Returns
    -------
    arg_type : {"scalar", "diag", "full", None}
        A string describing the argument type, else `None`
        if this is not possible
    """
    if isinstance(arg, types.Number):
        return "scalar"
    elif isinstance(arg, types.BaseTuple):
        if len(arg) == 2:
            return "diag"
        elif len(arg) == 4:
            return "full"

    return None


def term_mul(lhs, rhs):
    """
    Parameters
    ----------
    lhs : :class:`numba.Type`
    rhs : :class:`numba.Type`

    Returns
    -------
    multiplier : callable
        Function multiplying arguments of types lhs and rhs together
    """
    lhs_type = classify_arg(lhs)
    rhs_type = classify_arg(rhs)

    try:
        return _jones_typ_map[(lhs_type, rhs_type)]
    except KeyError:
        raise errors.TypingError(f"No known multiplication "
                                 f"function for {lhs} and {rhs}")


def unify_jones_terms(typingctx, lhs, rhs):
    """
    Unify Jones Term Types.
    """
    lhs_type = classify_arg(lhs)
    rhs_type = classify_arg(rhs)

    corr_map = {"scalar": 1, "diag": 2, "full": 4}

    try:
        lhs_corrs = corr_map[lhs_type]
        rhs_corrs = corr_map[rhs_type]
    except KeyError:
        raise errors.TypingError(f"{lhs} or {rhs} has no "
                                 f"entry in the {corr_map} "
                                 f"mapping")

    lhs_types = (lhs,) if lhs_corrs == 1 else tuple(lhs)
    rhs_types = (rhs,) if rhs_corrs == 1 else tuple(rhs)

    out_type = typingctx.unify_types(*lhs_types, *rhs_types)
    out_corrs = max(lhs_corrs, rhs_corrs)

    return out_type if out_corrs == 1 else types.Tuple((out_type,)*out_corrs)


@intrinsic
def tuple_adder(typingctx, t1, t2):
    if not isinstance(t1, types.BaseTuple):
        raise errors.TypingError(f"{t1} must be a Tuple")

    if not isinstance(t2, types.BaseTuple):
        raise errors.TypingError(f"{t2} must be a Tuple")

    if not len(t1) == len(t2):
        raise errors.TypeError(f"len({t1}) != len({t2})")

    sig = t1(t1, t2)

    def codegen(context, builder, signature, args):
        def _add(x, y):
            return x + y

        [t1, t2] = args
        [t1_type, t2_type] = signature.args
        return_type = signature.return_type

        llvm_ret_type = context.get_value_type(return_type)
        ret_tuple = cgutils.get_null_value(llvm_ret_type)

        for i, (t1e, t2e) in enumerate(zip(t1_type, t2_type)):
            v1 = builder.extract_value(t1, i)
            v2 = builder.extract_value(t2, i)
            vr = typingctx.unify_types(t1e, t2e)

            data = context.compile_internal(builder, _add,
                                            vr(t1e, t2e), [v1, v2])

            ret_tuple = builder.insert_value(ret_tuple, data, i)

        return ret_tuple

    return sig, codegen


def term_factory(args, kwargs, terms):
    term_arg_types = []
    term_arg_index = []
    term_kw_types = []
    term_kw_index = []

    term_state_types = []
    constructors = []

    for term in terms:
        sig = inspect.signature(term.term_type)
        wsig = SignatureAdapter(sig)

        arg_types, arg_i = zip(*(args[a] for a in wsig.args))
        kw_res = {k: kwargs.get(k, (types.Omitted(d), -1))
                  for k, d in wsig.kwargs.items()}

        kw_types = {k: v[0] for k, v in kw_res.items()}
        kw_i = {k: v[1] for k, v in kw_res.items()}

        term_arg_types.append(arg_types)
        term_arg_index.append(arg_i)
        term_kw_types.append(kw_types)
        term_kw_index.append(kw_i)

        term_state_types.append(term.term_type(*arg_types, **kw_types))

        init_sig = inspect.signature(term.initialiser)

        if init_sig != sig:
            raise ValueError(f"Initialiser signatures don't match "
                             f"{term.initialiser.__name__}{init_sig} vs "
                             f"initialiser{sig}")

        constructor = term.initialiser(*arg_types, **kw_types)
        constructor_sig = inspect.signature(constructor)

        if constructor_sig != sig:
            raise ValueError(f"Constructor signatures don't match "
                             f"{constructor.__name__}{init_sig} vs "
                             f"initialiser{sig}")

        constructors.append(constructor)

    @intrinsic
    def construct_terms(typingctx, args):
        if not isinstance(args, types.Tuple):
            raise errors.TypingError("args must be a Tuple")

        return_type = types.Tuple(term_state_types)
        sig = return_type(args)

        def codegen(context, builder, signature, args):
            typingctx = context.typing_context
            rvt = typingctx.resolve_value_type_prefer_literal
            return_type = signature.return_type

            if not isinstance(return_type, types.Tuple):
                raise errors.TypingError(
                    "signature.return_type should be a Tuple")

            llvm_ret_type = context.get_value_type(return_type)
            ret_tuple = cgutils.get_null_value(llvm_ret_type)

            if not len(args) == 1:
                raise errors.TypingError("args must contain a single value")

            constructor_args = []
            constructor_types = []

            # Our single argument is a tuple of arguments, but we
            # need to extract those arguments necessary to construct
            # the term StructRef
            it = zip(terms, term_arg_index, term_arg_types,
                     term_kw_index, term_kw_types)

            for term, arg_index, arg_types, kw_index, kw_types in it:
                cargs = [builder.extract_value(args[0], j) for j in arg_index]
                ctypes = list(arg_types)

                pysig = SignatureAdapter(inspect.signature(term.term_type))

                for k in pysig.kwargs:
                    kt = kw_types[k]
                    ki = kw_index[k]

                    if ki == -1:
                        assert isinstance(kt, types.Omitted)
                        value_type = rvt(kt.value)
                        const = context.get_constant_generic(
                            builder, value_type, kt.value)
                        cargs.append(const)
                        ctypes.append(value_type)
                    else:
                        cargs.append(builder.extract_value(args[0], ki))
                        ctypes.append(kt)

                constructor_args.append(cargs)
                constructor_types.append(ctypes)

            for ti in range(return_type.count):
                constructor_sig = return_type[ti](*constructor_types[ti])
                data = context.compile_internal(builder,
                                                constructors[ti],
                                                constructor_sig,
                                                constructor_args[ti])

                ret_tuple = builder.insert_value(ret_tuple, data, ti)

            return ret_tuple

        return sig, codegen

    samplers = [term.sampler() for term in terms]
    nterms = len(terms)

    def term_sampler(typingctx, state, s, r, t, a1, a2, c):
        if not isinstance(state, types.Tuple):
            raise errors.TypingError(f"{state} must be a Tuple")

        if not all(isinstance(t, TermStructRef) for t in state):
            raise errors.TypingError("All args must be an instance of Term")

        if not nterms == len(state) == len(term_state_types):
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
                f"(2) a Tuple containing 2 scalar correlations\n"
                f"(3) a Tuple containing 4 scalar correlations\n"
                f"but instead got a {typ}")

            if isinstance(typ, types.BaseTuple):
                if len(typ) not in (2, 4):
                    raise err

                if not all(isinstance(e, types.Number) for e in typ):
                    raise err

                continue

            raise err

        sampler_ret_type = sampler_return_types[0]

        for typ in sampler_return_types[1:]:
            sampler_ret_type = unify_jones_terms(typingctx,
                                                 sampler_ret_type, typ)

        sig = sampler_ret_type(state, s, r, t, a1, a2, c)

        def codegen(context, builder, signature, args):
            [state, s, r, t, a1, a2, c] = args
            jones = []

            for ti in range(nterms):
                sampling_fn = samplers[ti]

                # Build signature for the sampling function
                ret_type = sampler_return_types[ti]
                sampler_arg_types = ((term_state_types[ti],) +
                                     signature.args[1:])
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

            prev = jones[0]
            prev_t = sampler_return_types[0]

            for jrt, j in zip(sampler_return_types[1:], jones[1:]):
                jones_mul = term_mul(prev_t, jrt)
                jones_mul_typ = unify_jones_terms(context.typing_context,
                                                  prev_t, jrt)
                jones_sig = jones_mul_typ(prev_t, jrt)
                prev = context.compile_internal(builder, jones_mul,
                                                jones_sig,
                                                [prev, j])

                prev_t = jones_mul_typ

            return prev

        return sig, codegen

    @intrinsic
    def zero_vis(typingctx, state):
        # Figure out the jones type by calling term_sampler
        sig, _ = term_sampler(typingctx, state,
                              types.int64, types.int64,
                              types.int64, types.int64,
                              types.int64, types.int64)

        jones_type = sig.return_type

        if not isinstance(jones_type, types.BaseTuple):
            raise errors.TypingError(f"{jones_type} must be "
                                     f"a Tuple of numbers")

        if not all(isinstance(typ, types.Number) for typ in jones_type):
            raise errors.TypingError(f"{jones_type} must be "
                                     f"a Tuple of numbers")

        sig = jones_type(state)

        def codegen(context, builder, signature, args):
            jones_type = signature.return_type
            llvm_ret_type = context.get_value_type(jones_type)
            ret_tuple = cgutils.get_null_value(llvm_ret_type)

            for i, value_type in enumerate(jones_type):
                const = context.get_constant_generic(builder, value_type, 0)
                ret_tuple = builder.insert_value(ret_tuple, const, i)

            return ret_tuple

        return sig, codegen

    sample_terms = intrinsic(term_sampler)

    @register_jitable
    def pairwise_sampler(state, nsources, r, t, a1, a2, c):
        """
        This code based on https://github.com/numpy/numpy/pull/3685
        """
        X = zero_vis(state)
        stack = [(0, nsources)]

        while len(stack) > 0:
            start, end = stack.pop()
            nsrc = end - start

            if nsrc < 8:
                for s in range(start, start + end):
                    Y = sample_terms(state, s, r, t, a1, a2, c)
                    X = tuple_adder(X, Y)

            elif nsrc <= PAIRWISE_BLOCKSIZE:
                X0 = sample_terms(state, start + 0, r, t, a1, a2, c)
                X1 = sample_terms(state, start + 1, r, t, a1, a2, c)
                X2 = sample_terms(state, start + 2, r, t, a1, a2, c)
                X3 = sample_terms(state, start + 3, r, t, a1, a2, c)
                X4 = sample_terms(state, start + 4, r, t, a1, a2, c)
                X5 = sample_terms(state, start + 5, r, t, a1, a2, c)
                X6 = sample_terms(state, start + 6, r, t, a1, a2, c)
                X7 = sample_terms(state, start + 7, r, t, a1, a2, c)

                for s in range(8, nsrc - (nsrc % 8), 8):
                    o = start + s
                    Y0 = sample_terms(state, o + 0, r, t, a1, a2, c)
                    Y1 = sample_terms(state, o + 1, r, t, a1, a2, c)
                    Y2 = sample_terms(state, o + 2, r, t, a1, a2, c)
                    Y3 = sample_terms(state, o + 3, r, t, a1, a2, c)
                    Y4 = sample_terms(state, o + 4, r, t, a1, a2, c)
                    Y5 = sample_terms(state, o + 5, r, t, a1, a2, c)
                    Y6 = sample_terms(state, o + 6, r, t, a1, a2, c)
                    Y7 = sample_terms(state, o + 7, r, t, a1, a2, c)

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

                for o in range(s + 8, end):
                    Y = sample_terms(state, o, r, t, a1, a2, c)
                    X = tuple_adder(X, Y)
            else:
                ns2 = (nsrc // 2) - (nsrc % 8)
                stack.append((start, ns2))
                stack.append((start + ns2, end - ns2))

        return X

    return construct_terms, pairwise_sampler
