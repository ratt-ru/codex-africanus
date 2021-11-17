class Node:
    def __init__(self, neighbours=()):
        self._neighbours = neighbours

    def neighbours(self):
        return self._neighbours


class ArgumentGraph:
    def __init__(self, arg_name):
        self.value = arg_name


def pack_arguments(arg_names, args, terms, transformers):
    from collections import defaultdict
    from numba.core import types
    from numba.extending import intrinsic

    assert len(arg_names) == len(args[0])
    assert all(isinstance(n, types.Literal) for n in arg_names)
    assert all(n.literal_type is types.unicode_type for n in arg_names)

    names = tuple(n.literal_value for n in arg_names)
    desired = defaultdict(list)
    optional = defaultdict(list)
    maybe_create = defaultdict(list)

    for t in terms:
        for a in t.ARGS:
            desired[a].append(t)
        for k, d in t.KWARGS.items():
            optional[k].append((t, d))

    for t in transformers:
        for o in t.OUTPUTS:
            maybe_create[o].append(t)

    missing = set(desired.keys()) - set(names)
    available_args = set(names) | set(optional.keys())
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
                failed_transforms[arg].append((transformer, set(transformer.ARGS)))
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
                                f"{needed - set(names)} is missing "
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
            raise ValueError(f"Multiple terms: {terms} have contradicting "
                                f"definitions for {k}: {defaults}")                

        opt_defaults[k] = defaults.pop()        

    for name in names:
        opt_defaults.pop(name, None)

    print(f"supplied: {set(names)}, create: {can_create}, "
          f"optional: {opt_defaults}")

    from africanus.rime.monolithic.argpack import ArgumentPack
    from numba.core import cgutils, typing, errors, compiler
    from numba.core.typed_passes import type_inference_stage

    @intrinsic
    def pack_arguments(typingctx, args):
        assert len(args) == len(names)
        it = zip(names, args, range(len(names)))
        arg_info = {n: (t, i) for n, t, i in it}

        rvt = typingctx.resolve_value_type_prefer_literal
        optionals = [(n, rvt(d), d) for n, d in opt_defaults.items()]
        opt_return_types = tuple(p[1] for p in optionals)

        transform_return_types = []

        for name, transformer in can_create.items():
            transform = transformer.transform()
            ir = compiler.run_frontend(transform)
            arg_types = tuple(arg_info[a][0] for a in transformer.ARGS)                
            type_infer = type_inference_stage(typingctx, ir, arg_types, None)
            return_type = type_infer.return_type

            if len(transformer.OUTPUTS) == 0:
                raise errors.TypingError(
                    f"{transformer} produces no outputs")
            elif len(transformer.OUTPUTS) > 1:
                if not isinstance(return_type, types.Tuple):
                    raise errors.TypingError(
                        f"{transformer} produces {transformer.OUTPUTS} "
                        f"but {transformer}.transform does not return "
                        f"a tuple of these ouputs, but {return_type}")

                if len(transformer.OUTPUTS) != len(return_type):
                    raise errors.TypingError(
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

            for i, typ in enumerate(signature.args[0]):
                value = builder.extract_value(args[0], i)
                context.nrt.incref(builder, signature.args[0][i], value)
                ret_tuple = builder.insert_value(ret_tuple, value, i)

            n = len(names)

            for i, (_, typ, default) in enumerate(optionals):
                value = context.get_constant_generic(builder, typ, default)
                ret_tuple = builder.insert_value(ret_tuple, value, i + n)

            n += len(optionals)

            for i, (_, transformer) in enumerate(can_create.items()):
                transform_args = []
                transform_types = []
                transform_fn = transformer.transform()

                for name in transformer.ARGS:
                    try:
                        typ, j  = arg_info[name]
                    except KeyError:
                        raise errors.TypingError(f"{name} is not present in arg_types")

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
    