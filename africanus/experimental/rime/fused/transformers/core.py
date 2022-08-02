import inspect


from africanus.experimental.rime.fused.error import InvalidSignature


def sigcheck_factory(expected_sig):
    def check_transformer_sig(self, fn):
        sig = inspect.signature(fn)
        if sig != expected_sig:
            raise ValueError(f"{fn.__name__}{sig} should be "
                             f"{fn.__name__}{expected_sig}")

    return check_transformer_sig


class TransformerMetaClass(type):
    """
    Metaclass which checks that the appropriate methods are
    implemented on any subclass of `Transformer` and that their
    signatures agree with each other.

    Also sets `ARGS`, `KWARGS` and `ALL_ARGS`
    class members on the subclass based on the above
    signatures
    """
    REQUIRED = ("dask_schema", "init_fields")

    @classmethod
    def _expand_namespace(cls, name, namespace):
        methods = []

        for method_name in cls.REQUIRED:
            try:
                method = namespace[method_name]
            except KeyError:
                raise NotImplementedError(f"{name}.{method_name}")
            else:
                methods.append(method)

        methods = dict(zip(cls.REQUIRED, methods))
        init_fields_sig = inspect.signature(methods["init_fields"])
        field_params = list(init_fields_sig.parameters.values())

        if len(init_fields_sig.parameters) < 2:
            raise InvalidSignature(f"{name}.init_fields{init_fields_sig} "
                                   f"should be "
                                   f"{name}.init_fields(self, typingctx, ...)")

        it = iter(init_fields_sig.parameters.items())
        first, second = next(it), next(it)

        if first[0] != "self" or second[0] != "typingctx":
            raise InvalidSignature(f"{name}.init_fields{init_fields_sig} "
                                   f"should be "
                                   f"{name}.init_fields(self, typingctx, ...)")

        for n, p in it:
            if p.kind == p.VAR_POSITIONAL:
                raise InvalidSignature(f"*{n} in "
                                       f"{name}.init_fields{init_fields_sig} "
                                       f"is not supported")

            if p.kind == p.VAR_KEYWORD:
                raise InvalidSignature(f"**{n} in "
                                       f"{name}.init_fields{init_fields_sig} "
                                       f"is not supported")

        dask_schema_sig = inspect.signature(methods["dask_schema"])
        expected_dask_params = field_params[0:1] + field_params[2:]
        expected_dask_sig = init_fields_sig.replace(
            parameters=expected_dask_params)

        if dask_schema_sig != expected_dask_sig:
            raise InvalidSignature(f"{name}.dask_schema{dask_schema_sig} "
                                   f"should be "
                                   f"{name}.dask_schema{expected_dask_sig}")

        if not ("OUTPUTS" in namespace and
                isinstance(namespace["OUTPUTS"], (tuple, list)) and
                all(isinstance(o, str) for o in namespace["OUTPUTS"])):

            raise InvalidSignature(f"{name}.OUTPUTS should be a tuple "
                                   f"of the names of the outputs produced "
                                   f"by this transformer")

        transform_sig = init_fields_sig.replace(parameters=field_params[2:])

        namespace["OUTPUTS"] = tuple(namespace["OUTPUTS"])

        args = tuple(n for n, p in init_fields_sig.parameters.items()
                     if p.kind in {p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD}
                     and n not in {"self", "typingctx"}
                     and p.default is p.empty)

        kw = ((n, p.default) for n, p in init_fields_sig.parameters.items()
              if p.kind in {p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY}
              and n not in {"self", "typingctx"}
              and p.default is not p.empty)

        namespace = namespace.copy()
        namespace["ARGS"] = args
        namespace["KWARGS"] = dict(kw)
        namespace["ALL_ARGS"] = args + tuple(k for k, _ in kw)
        namespace["transform_validator"] = sigcheck_factory(transform_sig)

        return namespace

    @classmethod
    def transformer_in_bases(cls, bases):
        """ Is `Transformer` in bases? """
        for base in bases:
            if base is Transformer or cls.transformer_in_bases(base.__bases__):
                return True

        return False

    def __new__(mcls, name, bases, namespace):
        # Check methods on any subclasses of Transformer
        # and expand the subclass namespace
        if mcls.transformer_in_bases(bases):
            namespace = mcls._expand_namespace(name, namespace)

        return super(TransformerMetaClass, mcls).__new__(
            mcls, name, bases, namespace)


class Transformer(metaclass=TransformerMetaClass):
    def __init__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__class__.__name__
