from collections import defaultdict

from numba.core import compiler, cgutils, types
from numba.core.errors import TypingError
from numba.core.extending import intrinsic
from numba.experimental import structref
from numba.core.typed_passes import type_inference_stage

from africanus.rime.monolithic.argpack import ArgumentPack
from africanus.rime.monolithic.terms.core import StateStructRef

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
        raise TypingError(f"No known multiplication "
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
        raise TypingError(f"{lhs} or {rhs} has no "
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
        raise TypingError(f"{t1} must be a Tuple")

    if not isinstance(t2, types.BaseTuple):
        raise TypingError(f"{t2} must be a Tuple")

    if not len(t1) == len(t2):
        raise TypingError(f"len({t1}) != len({t2})")

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


class IntrinsicFactory:
    def __init__(self, arg_names, terms, transformers):
        self.names = arg_names
        self.terms = terms
        self.transformers = transformers
        od, cc = self._resolve_arg_dependencies()
        self.optional_defaults = od
        self.can_create = cc
        self.output_names = (self.names +
                             tuple(self.optional_defaults.keys()) +
                             tuple(self.can_create.keys()))

    def _resolve_arg_dependencies(self):
        desired = defaultdict(list)
        optional = defaultdict(list)
        maybe_create = defaultdict(list)

        for t in self.terms:
            for a in t.ARGS:
                desired[a].append(t)
            for k, d in t.KWARGS.items():
                optional[k].append((t, d))

        for t in self.transformers:
            for o in t.OUTPUTS:
                maybe_create[o].append(t)

        missing = set(desired.keys()) - set(self.names)
        available_args = set(self.names) | set(optional.keys())
        failed_transforms = defaultdict(list)
        can_create = {}

        # Try create missing argument with transformers
        for arg in list(missing):
            # We already know how to create it
            if arg in can_create:
                continue

            # We don't know how to create
            if arg not in maybe_create:
                continue

            for transformer in maybe_create[arg]:
                # We didn't have the arguments, make a note of this
                if not set(transformer.ARGS).issubset(available_args):
                    failed_transforms[arg].append(
                        (transformer, set(transformer.ARGS)))
                    continue

            # The transformer can create arg
            if arg not in failed_transforms:
                can_create[arg] = transformer
                missing.remove(arg)

        # Fail if required arguments are missing
        for arg in missing:
            terms_wanting = desired[arg]
            err_msgs = []
            err_msgs.append(f"{set(terms_wanting)} need(s) '{arg}'.")

            if arg in failed_transforms:
                for transformer, needed in failed_transforms[arg]:
                    err_msgs.append(f"{transformer} can create {arg} "
                                    f"but needs {needed}, of which "
                                    f"{needed - set(self.names)} is missing "
                                    f"from the input arguments.")

            raise ValueError("\n".join(err_msgs))

        opt_defaults = {}

        for transformer in can_create.values():
            for k, d in transformer.KWARGS.items():
                optional[k].append((transformer, d))

        for k, v in optional.items():
            _, defaults = zip(*v)
            defaults = set(defaults)

            if len(defaults) != 1:
                raise ValueError(f"Multiple terms: {self.terms} have "
                                 f"contradicting definitions for "
                                 f"{k}: {defaults}")

            opt_defaults[k] = defaults.pop()

        for name in self.names:
            opt_defaults.pop(name, None)

        return opt_defaults, can_create

    def pack_argument_fn(self):
        @intrinsic
        def pack_arguments(typingctx, args):
            assert len(args) == len(self.names)
            it = zip(self.names, args, range(len(self.names)))
            arg_info = {n: (t, i) for n, t, i in it}

            rvt = typingctx.resolve_value_type_prefer_literal
            optionals = [(n, rvt(d), d) for n, d
                         in self.optional_defaults.items()]
            opt_return_types = tuple(p[1] for p in optionals)

            transform_return_types = []

            for name, transformer in self.can_create.items():
                transform = transformer.transform()
                ir = compiler.run_frontend(transform)
                arg_types = tuple(arg_info[a][0] for a in transformer.ARGS)
                type_infer = type_inference_stage(
                    typingctx, ir, arg_types, None)
                return_type = type_infer.return_type

                if len(transformer.OUTPUTS) == 0:
                    raise TypingError(f"{transformer} produces no outputs")
                elif len(transformer.OUTPUTS) > 1:
                    if not isinstance(return_type, types.Tuple):
                        raise TypingError(
                            f"{transformer} produces {transformer.OUTPUTS} "
                            f"but {transformer}.transform does not return "
                            f"a tuple of these ouputs, but {return_type}")

                    if len(transformer.OUTPUTS) != len(return_type):
                        raise TypingError(
                            f"{transformer} produces {transformer.OUTPUTS} "
                            f"but {transformer}.transform does not return "
                            f"a tuple of the same length, but {return_type}")

                transform_return_types.append(return_type)

            return_type = types.Tuple(args.types +
                                      opt_return_types +
                                      tuple(transform_return_types))
            sig = return_type(args)

            def codegen(context, builder, signature, args):
                return_type = signature.return_type
                llvm_ret_type = context.get_value_type(return_type)
                ret_tuple = cgutils.get_null_value(llvm_ret_type)

                # Extract supplied arguments from original arg tuple
                # and insert into the new one
                for i, typ in enumerate(signature.args[0]):
                    value = builder.extract_value(args[0], i)
                    context.nrt.incref(builder, signature.args[0][i], value)
                    ret_tuple = builder.insert_value(ret_tuple, value, i)

                n = len(self.names)

                # Insert necessary optional defaults (kwargs) into the
                # new argument tuple
                for i, (name, typ, default) in enumerate(optionals):
                    if name != self.output_names[i + n]:
                        raise TypingError(
                            f"{name} != {self.output_names[i + n]}")

                    value = context.get_constant_generic(builder, typ, default)
                    ret_tuple = builder.insert_value(ret_tuple, value, i + n)

                n += len(optionals)

                # Apply any argument transforms and insert their results
                # into the new argument tuple
                for i, (v, transformer) in enumerate(self.can_create.items()):
                    if v != self.output_names[i + n]:
                        raise TypingError(
                            f"{v} != {self.output_names[i + n]}")

                    transform_args = []
                    transform_types = []
                    transform_fn = transformer.transform()

                    for name in transformer.ARGS:
                        try:
                            typ, j = arg_info[name]
                        except KeyError:
                            raise TypingError(
                                f"{name} is not present in arg_types")

                        value = builder.extract_value(args[0], j)
                        context.nrt.incref(builder, typ, value)

                        transform_args.append(value)
                        transform_types.append(typ)

                    ret_type = transform_return_types[i]
                    transform_sig = ret_type(*transform_types)

                    value = context.compile_internal(builder,  # noqa
                                                     transform_fn,
                                                     transform_sig,
                                                     transform_args)

                    ret_tuple = builder.insert_value(ret_tuple, value, i + n)

                return ret_tuple

            return sig, codegen

        return pack_arguments

    def term_state_fn(self):
        @intrinsic
        def term_state(typingctx, args):
            if not isinstance(args, types.Tuple):
                raise TypingError(f"args must be a Tuple but is {args}")

            arg_pack = ArgumentPack(
                self.output_names, args, tuple(range(len(args))))

            constructors = []
            state_fields = []

            # Query Terms for fields and their associated types
            # that should be created on the State object
            for term in self.terms:
                arg_idx = arg_pack.indices(*term.ALL_ARGS)
                arg_types = {a: args[i]
                             for a, i in zip(term.ALL_ARGS, arg_idx)}
                state_fields.extend(term.fields(**arg_types))

            # Now define all fields for the State type
            arg_fields = [(k, args[i]) for k, (_, i) in arg_pack.items()]
            state_type = StateStructRef(arg_fields + state_fields)

            for term in self.terms:
                arg_idx = arg_pack.indices(*term.ALL_ARGS)
                arg_types = {a: args[i]
                             for a, i in zip(term.ALL_ARGS, arg_idx)}
                init_types = {"state": state_type, **arg_types}
                constructor = term.initialiser(**init_types)
                term.validate_constructor(constructor)
                constructors.append(constructor)

            sig = state_type(args)

            def codegen(context, builder, signature, args):
                if not len(args) == 1:
                    raise TypingError("args must contain a single value")

                typingctx = context.typing_context
                rvt = typingctx.resolve_value_type_prefer_literal

                def make_struct():
                    """ Allocate the structure """
                    return structref.new(state_type)

                state = context.compile_internal(builder, make_struct,
                                                 state_type(), [])
                U = structref._Utils(context, builder, state_type)
                data_struct = U.get_data_struct(state)

                for arg_name, (_, i) in arg_pack.items():
                    value = builder.extract_value(args[0], i)
                    value_type = signature.args[0][i]
                    context.nrt.incref(builder, value_type, value)
                    field_type = state_type.field_dict[arg_name]
                    casted = context.cast(builder, value,
                                          value_type, field_type)
                    old_value = getattr(data_struct, arg_name)
                    context.nrt.incref(builder, value_type, casted)
                    context.nrt.decref(builder, value_type, old_value)
                    setattr(data_struct, arg_name, casted)

                constructor_args = []
                constructor_types = []

                # Our single argument is a tuple of arguments, but we
                # need to extract those arguments necessary to construct
                # the term StructRef
                for term in self.terms:
                    cargs = [state]
                    ctypes = [state_type]

                    arg_types = arg_pack.types(*term.ALL_ARGS)
                    arg_index = arg_pack.indices(*term.ALL_ARGS)

                    for typ, i in zip(arg_types, arg_index):
                        if isinstance(typ, types.Omitted):
                            const_type = rvt(typ.value)
                            const = context.get_constant_generic(
                                builder, const_type, typ.value)
                            cargs.append(const)
                            ctypes.append(const_type)
                        else:
                            assert not isinstance(typ, types.Omitted)
                            assert i != -1
                            cargs.append(builder.extract_value(args[0], i))
                            ctypes.append(typ)

                    constructor_args.append(cargs)
                    constructor_types.append(ctypes)

                for ti in range(len(self.terms)):
                    constructor_sig = types.none(*constructor_types[ti])
                    context.compile_internal(builder,
                                             constructors[ti],
                                             constructor_sig,
                                             constructor_args[ti])

                return state

            return sig, codegen

        return term_state

    def term_sampler_fn(self):
        samplers = [term.sampler() for term in self.terms]

        for term, sampler in zip(self.terms, samplers):
            term.validate_sampler(sampler)

        nterms = len(self.terms)

        @intrinsic
        def term_sampler(typingctx, state, s, r, t, a1, a2, c):
            if not isinstance(state, StateStructRef):
                raise TypingError(f"{state} must be a StateStructRef")

            sampler_ir = list(map(compiler.run_frontend, samplers))
            idx_types = (s, r, t, a1, a2, c)
            ir_args = (state,) + idx_types
            type_infer = [type_inference_stage(typingctx, ir, ir_args, None)
                          for ir in sampler_ir]
            sampler_return_types = [ti.return_type for ti in type_infer]

            # Sanity check the sampler return types
            for typ, sampler in zip(sampler_return_types, samplers):
                if isinstance(typ, types.Number):
                    continue

                err = TypingError(
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
                [state_type, _, _, _, _, _, _] = signature.args
                jones = []

                for ti in range(nterms):
                    sampling_fn = samplers[ti]

                    # Build signature for the sampling function
                    ret_type = sampler_return_types[ti]
                    sampler_arg_types = (state_type,) + signature.args[1:]
                    sampler_sig = ret_type(*sampler_arg_types)

                    # Build LLVM arguments for the sampling function
                    sampler_args = [state, s, r, t, a1, a2, c]

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

        return term_sampler
