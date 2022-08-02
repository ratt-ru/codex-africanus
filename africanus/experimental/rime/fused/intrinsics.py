from collections import defaultdict
from distutils.version import LooseVersion
from functools import partial

import numba
from numba.core import compiler, cgutils, types
from numba.core.errors import TypingError
from numba.core.extending import intrinsic
from numba.experimental import structref
from numba.core.typed_passes import type_inference_stage

import numpy as np

from africanus.averaging.support import _unique_internal

from africanus.experimental.rime.fused.arguments import ArgumentPack
from africanus.experimental.rime.fused.terms.core import StateStructRef

try:
    NUMBA_MAJOR, NUMBA_MINOR, _ = LooseVersion(numba.__version__).version
except AttributeError:
    # Readthedocs
    NUMBA_MAJOR, NUMBA_MINOR = 0, 0


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
    return (
        lhs[0]*rhs,
        lhs[1]*rhs,
        lhs[2]*rhs,
        lhs[3]*rhs)


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


def hermitian_scalar(jones):
    return np.conj(jones)


def hermitian_diag(jones):
    return (np.conj(jones[0]), np.conj(jones[1]))


def hermitian_full(jones):
    return (np.conj(jones[0]),
            np.conj(jones[2]),
            np.conj(jones[1]),
            np.conj(jones[3]))


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


_hermitian_map = {
    "scalar": hermitian_scalar,
    "diag": hermitian_diag,
    "full": hermitian_full
}


def hermitian(jones):
    jones_type = classify_arg(jones)

    try:
        return _hermitian_map[jones_type]
    except KeyError:
        raise TypingError(f"No known hermitian function "
                          f"for {jones}: {jones_type}.")


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
    KEY_ARGS = ("utime", "time_index",
                "uantenna", "antenna1_index", "antenna2_index",
                "ufeed", "feed1_index", "feed2_index")

    def __init__(self, arg_dependencies):
        self.argdeps = arg_dependencies

    def _resolve_arg_dependencies(self):
        argdeps = self.argdeps

        # KEY_ARGS will be created
        supplied_args = set(argdeps.names) | set(self.KEY_ARGS)
        missing = set(argdeps.desired.keys()) - supplied_args
        available_args = set(argdeps.names) | supplied_args
        failed_transforms = defaultdict(list)
        can_create = {}

        # Try create missing argument with transformers
        for arg in list(missing):
            # We already know how to create it
            if arg in can_create:
                continue

            # We don't know how to create
            if arg not in argdeps.maybe_create:
                continue

            for transformer in argdeps.maybe_create[arg]:
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
            terms_wanting = argdeps.desired[arg]
            err_msgs = []
            err_msgs.append(f"{set(terms_wanting)} need(s) '{arg}'.")

            if arg in failed_transforms:
                for transformer, needed in failed_transforms[arg]:
                    err_msgs.append(f"{transformer} can create {arg} "
                                    f"but needs {needed}, of which "
                                    f"{needed - set(argdeps.names)} is "
                                    f"missing from the input arguments.")

            raise ValueError("\n".join(err_msgs))

        opt_defaults = {}

        for transformer in can_create.values():
            for k, d in transformer.KWARGS.items():
                argdeps.optional[k].append((transformer, d))

        for k, v in argdeps.optional.items():
            _, defaults = zip(*v)
            defaults = set(defaults)

            if len(defaults) != 1:
                raise ValueError(f"Multiple terms: {argdeps.terms} have "
                                 f"contradicting definitions for "
                                 f"{k}: {defaults}")

            opt_defaults[k] = defaults.pop()

        for name in argdeps.names:
            opt_defaults.pop(name, None)

        return opt_defaults, can_create

    def pack_optionals_and_indices_fn(self):
        argdeps = self.argdeps
        out_names = (argdeps.names +
                     tuple(argdeps.optional_defaults.keys()) +
                     tuple(argdeps.KEY_ARGS))

        @intrinsic
        def pack_index(typingctx, args):
            assert len(args) == len(argdeps.names)
            it = zip(argdeps.names, args, range(len(argdeps.names)))
            arg_info = {n: (t, i) for n, t, i in it}

            key_types = {
                "utime": arg_info["time"][0],
                "time_index": types.int64[:],
                "uantenna": arg_info["antenna1"][0],
                "antenna1_index": types.int64[:],
                "antenna2_index": types.int64[:],
                "ufeed": arg_info["feed1"][0],
                "feed1_index": types.int64[:],
                "feed2_index": types.int64[:]
            }

            if tuple(key_types.keys()) != argdeps.KEY_ARGS:
                raise RuntimeError(
                    f"{tuple(key_types.keys())} != {argdeps.KEY_ARGS}")

            rvt = typingctx.resolve_value_type_prefer_literal
            optionals = [(n, rvt(d), d) for n, d
                         in argdeps.optional_defaults.items()]
            optional_types = tuple(p[1] for p in optionals)

            return_type = types.Tuple(args.types + optional_types +
                                      tuple(key_types.values()))
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

                n = len(signature.args[0])

                # Insert necessary optional defaults (kwargs) into the
                # new argument tuple
                for i, (name, typ, default) in enumerate(optionals):
                    if name != out_names[i + n]:
                        raise TypingError(f"{name} != {out_names[i + n]}")
                    value = context.get_constant_generic(builder, typ, default)
                    ret_tuple = builder.insert_value(ret_tuple, value, i + n)

                # Compute indexing arguments and insert into
                # the new tuple
                fn_args = [builder.extract_value(args[0], arg_info[a][1])
                           for a in argdeps.REQUIRED_ARGS]
                fn_arg_types = tuple(arg_info[k][0] for k
                                     in argdeps.REQUIRED_ARGS)
                fn_sig = types.Tuple(list(key_types.values()))(*fn_arg_types)

                def _indices(time, antenna1, antenna2, feed1, feed2):
                    utime, _, time_index, _ = _unique_internal(time)
                    uants = np.unique(np.concatenate((antenna1, antenna2)))
                    ufeeds = np.unique(np.concatenate((feed1, feed2)))
                    antenna1_index = np.searchsorted(uants, antenna1)
                    antenna2_index = np.searchsorted(uants, antenna2)
                    feed1_index = np.searchsorted(ufeeds, feed1)
                    feed2_index = np.searchsorted(ufeeds, feed2)

                    return (utime, time_index,
                            uants, antenna1_index, antenna2_index,
                            ufeeds, feed1_index, feed2_index)

                index = context.compile_internal(builder, _indices,
                                                 fn_sig, fn_args)

                n += len(optionals)

                for i, (name, value) in enumerate(key_types.items()):
                    if name != out_names[i + n]:
                        raise TypingError(f"{name} != {out_names[i + n]}")

                    value = builder.extract_value(index, i)
                    ret_tuple = builder.insert_value(ret_tuple, value, i + n)

                return ret_tuple

            return sig, codegen

        return out_names, pack_index

    def pack_transformed_fn(self, arg_names):
        argdeps = self.argdeps
        transformers = list(set(t for _, t in argdeps.can_create.items()))
        out_names = arg_names + tuple(o for t in transformers
                                      for o in t.OUTPUTS)

        @intrinsic
        def pack_transformed(typingctx, args):
            assert len(args) == len(arg_names)
            it = zip(arg_names, args, range(len(arg_names)))
            arg_info = {n: (t, i) for n, t, i in it}

            rvt = typingctx.resolve_value_type_prefer_literal
            transform_output_types = []

            for transformer in transformers:
                # Figure out argument types for calling init_fields
                kw = {}

                for a in transformer.ARGS:
                    kw[a] = arg_info[a][0]

                for a, d in transformer.KWARGS.items():
                    try:
                        kw[a] = arg_info[a][0]
                    except KeyError:
                        kw[a] = rvt(d)

                fields, _ = transformer.init_fields(typingctx, **kw)

                if len(transformer.OUTPUTS) == 0:
                    raise TypingError(f"{transformer} produces no outputs")
                elif len(transformer.OUTPUTS) > 1:
                    if len(transformer.OUTPUTS) != len(fields):
                        raise TypingError(
                            f"{transformer} produces {transformer.OUTPUTS} "
                            f"but {transformer}.init_fields does not return "
                            f"a tuple of the same length, but {fields}")

                transform_output_types.extend(t for _, t in fields)

            # Create a return tuple containing the existing arguments
            # with the transformed outputs added to the end
            return_type = types.Tuple(args.types +
                                      tuple(transform_output_types))

            # Sanity check
            if len(return_type) != len(out_names):
                raise TypingError(f"len(return_type): {len(return_type)} != "
                                  f"len(out_names): {len(out_names)}")

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

                # Apply any argument transforms and insert their results
                # into the new argument tuple
                n = len(signature.args[0])
                i = 0

                for transformer in transformers:
                    # Check that outputs line up with output names
                    for j, o in enumerate(transformer.OUTPUTS):
                        if o != out_names[i + j + n]:
                            raise TypingError(f"{o} != {out_names[i + j + n]}")

                    transform_args = []
                    transform_types = []

                    # Get required arguments out of the argument pack
                    for name in transformer.ARGS:
                        try:
                            typ, j = arg_info[name]
                        except KeyError:
                            raise TypingError(
                                f"{name} is not present in arg_types")

                        value = builder.extract_value(args[0], j)
                        transform_args.append(value)
                        transform_types.append(typ)

                    # Generate defaults
                    for name, default in transformer.KWARGS.items():
                        default_typ = rvt(default)
                        default_value = context.get_constant_generic(
                                                builder,
                                                default_typ,
                                                default)

                        transform_types.append(default_typ)
                        transform_args.append(default_value)

                    # Get the transformer fields and function
                    transform_fields, transform_fn = transformer.init_fields(
                        typingctx, *transform_types)

                    single_return = len(transform_fields) == 1

                    # Handle singleton vs tuple return types
                    if single_return:
                        ret_type = transform_fields[0][1]
                    else:
                        typs = [t for _, t in transform_fields]
                        ret_type = types.Tuple(typs)

                    # Call the transform function
                    transform_sig = ret_type(*transform_types)
                    value = context.compile_internal(builder,  # noqa
                                                     transform_fn,
                                                     transform_sig,
                                                     transform_args)

                    # Unpack the returned value and insert into
                    # return_tuple
                    if single_return:
                        ret_tuple = builder.insert_value(ret_tuple, value,
                                                         i + n)
                        i += 1
                    else:
                        for j, o in enumerate(transformer.OUTPUTS):
                            element = builder.extract_value(value, j)
                            ret_tuple = builder.insert_value(ret_tuple,
                                                             element,
                                                             i + n)
                            i += 1

                return ret_tuple

            return sig, codegen

        return out_names, pack_transformed

    def term_state_fn(self, arg_names):
        argdeps = self.argdeps

        @intrinsic
        def term_state(typingctx, args):
            if not isinstance(args, types.Tuple):
                raise TypingError(f"args must be a Tuple but is {args}")

            if len(arg_names) != len(args):
                raise TypingError(f"len(arg_names): {len(arg_names)} != "
                                  f"len(args): {len(args)}")

            arg_pack = ArgumentPack(arg_names, args, tuple(range(len(args))))

            state_fields = []
            term_fields = []
            constructors = []

            # Query Terms for fields and their associated types
            # that should be created on the State object
            for term in argdeps.terms:
                it = zip(term.ALL_ARGS, arg_pack.indices(*term.ALL_ARGS))
                arg_types = {a: args[i] for a, i in it}
                fields, constructor = term.init_fields(typingctx, **arg_types)
                term.validate_constructor(constructor)
                term_fields.append(fields)
                state_fields.extend(fields)
                constructors.append(constructor)

            # Now define all fields for the State type
            arg_fields = [(k, args[i]) for k, (_, i) in arg_pack.items()]
            state_type = StateStructRef(arg_fields + state_fields)
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
                    # We increment the reference count here
                    # as we're taking a reference from data in
                    # the args tuple and placing it on the structref
                    context.nrt.incref(builder, value_type, value)
                    field_type = state_type.field_dict[arg_name]
                    casted = context.cast(builder, value,
                                          value_type, field_type)
                    context.nrt.incref(builder, value_type, casted)

                    # The old value on the structref is being replaced,
                    # decrease it's reference count
                    old_value = getattr(data_struct, arg_name)
                    context.nrt.decref(builder, value_type, old_value)
                    setattr(data_struct, arg_name, casted)

                constructor_args = []
                constructor_types = []

                # Our single argument is a tuple of arguments, but we
                # need to extract those arguments necessary to construct
                # the term StructRef
                for term in argdeps.terms:
                    cargs = []
                    ctypes = []

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

                for ti in range(len(argdeps.terms)):
                    fields = term_fields[ti]
                    nfields = len(fields)

                    if nfields == 0:
                        return_type = types.none
                    elif nfields == 1:
                        return_type = fields[0][1]
                    else:
                        return_types = [f[1] for f in fields]
                        return_type = types.Tuple(return_types)

                    constructor_sig = return_type(*constructor_types[ti])
                    return_value = context.compile_internal(
                                    builder, constructors[ti],
                                    constructor_sig, constructor_args[ti])

                    if nfields == 0:
                        pass
                    elif nfields == 1:
                        arg_name, typ = fields[0]
                        old_value = getattr(data_struct, arg_name)
                        context.nrt.decref(builder, typ, old_value)
                        setattr(data_struct, arg_name, return_value)
                    else:
                        for i, (arg_name, typ) in enumerate(fields):
                            value = builder.extract_value(return_value, i)
                            context.nrt.incref(builder, typ, value)
                            old_value = getattr(data_struct, arg_name)
                            context.nrt.decref(builder, typ, old_value)
                            setattr(data_struct, arg_name, value)

                return state

            return sig, codegen

        return term_state

    def term_sampler_fn(self):
        argdeps = self.argdeps
        terms = argdeps.terms

        samplers = [term.sampler() for term in terms]

        for term, sampler in zip(terms, samplers):
            term.validate_sampler(sampler)

        nterms = len(terms)

        @intrinsic
        def term_sampler(typingctx, state, s, r, t, f1, f2, a1, a2, c):
            if not isinstance(state, StateStructRef):
                raise TypingError(f"{state} must be a StateStructRef")

            sampler_ir = list(map(compiler.run_frontend, samplers))
            ir_args = (state, s, r, t, f1, f2, a1, a2, c)

            if NUMBA_MAJOR > 0 or NUMBA_MINOR >= 54:
                # NOTE(sjperkins)
                # numba 0.54 wants a targetctx for type_inference_stage
                # Assume we're dealing with a CPU Target  in order to derive
                # the targetctx. This is a fair assumption given that we're
                # writing CPU intrinsics. Note that numba is also assuming
                # CPU Targets in their code base in 0.54, at least. Look for
                # the ability to figure out the current target context manager
                # in future releases in order to find a better solution here.
                from numba.core.registry import cpu_target
                if cpu_target.typing_context != typingctx:
                    raise TypingError("typingctx's don't match")

                tis = partial(type_inference_stage,
                              typingctx=typingctx,
                              targetctx=cpu_target.target_context,
                              args=ir_args,
                              return_type=None)
            else:
                tis = partial(type_inference_stage,
                              typingctx=typingctx,
                              args=ir_args,
                              return_type=None)

            type_infer = [tis(interp=ir) for ir in sampler_ir]
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

            sig = sampler_ret_type(state, s, r, t, f1, f2, a1, a2, c)

            def codegen(context, builder, signature, args):
                [state, s, r, t, f1, f2, a1, a2, c] = args
                [state_type, _, _, _, _, _, _, _, _] = signature.args
                jones = []

                for ti in range(nterms):
                    sampling_fn = samplers[ti]

                    # Build signature for the sampling function
                    ret_type = sampler_return_types[ti]
                    sampler_arg_types = (state_type,) + signature.args[1:]
                    sampler_sig = ret_type(*sampler_arg_types)

                    # Build LLVM arguments for the sampling function
                    sampler_args = [state, s, r, t, f1, f2, a1, a2, c]

                    # Call the sampling function
                    data = context.compile_internal(builder,  # noqa
                                                    sampling_fn,
                                                    sampler_sig,
                                                    sampler_args)

                    # Apply hermitian transform if this is a right term
                    if terms[ti].configuration == "right":
                        data = context.compile_internal(builder,  # noqa
                                                        hermitian(ret_type),
                                                        ret_type(ret_type),
                                                        [data])

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
