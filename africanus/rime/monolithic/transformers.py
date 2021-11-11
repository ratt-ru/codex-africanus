import inspect


from numba.core import types

from africanus.rime.monolithic.common import result_type
from africanus.rime.monolithic.error import InvalidSignature


class TransformerMetaClass(type):
    """
    Metaclass which checks that the appropriate methods are
    implemented on any subclass of `Transformer` and that their
    signatures agree with each other.

    Also sets `ARGS`, `KWARGS` and `ALL_ARGS`
    class members on the subclass based on the above
    signatures
    """
    REQUIRED = ("dask_schema", "fields", "transform")

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
        fields_sig = inspect.signature(methods["fields"])

        for i, (n, p) in enumerate(fields_sig.parameters.items()):
            if i == 0 and n != "self":
                raise InvalidSignature(f"{name}.fields{fields_sig} "
                                       f"should be "
                                       f"{name}.fields(self, ...)")

            if p.kind == p.VAR_POSITIONAL:
                raise InvalidSignature(f"*{n} in "
                                       f"{name}.fields{fields_sig} "
                                       f"is not supported")

            if p.kind == p.VAR_KEYWORD:
                raise InvalidSignature(f"**{n} in "
                                       f"{name}.fields{fields_sig} "
                                       f"is not supported")

        transform_sig = inspect.signature(methods["transform"])

        if fields_sig != transform_sig:
            raise InvalidSignature(f"{name}.transform{transform_sig} "
                                   f"should be "
                                   f"{name}.transform{fields_sig}")

        dask_schema_sig = inspect.signature(methods["dask_schema"])

        if fields_sig != dask_schema_sig:
            raise InvalidSignature(f"{name}.dask_schema{dask_schema_sig} "
                                   f"should be "
                                   f"{name}.dask_schema{fields_sig}")

        valid_outputs = "OUTPUTS" in namespace
        valid_outputs = (valid_outputs and
                         isinstance(namespace["OUTPUTS"], (tuple, list)))
        valid_outputs = (valid_outputs and
                         all(isinstance(o, str) for o in namespace["OUTPUTS"]))

        if not valid_outputs:
            raise InvalidSignature(f"{name}.OUTPUTS should be a tuple "
                                   f"of the names of the outputs produced "
                                   f"by this transformer")

        namespace["OUTPUTS"] = tuple(namespace["OUTPUTS"])

        args = tuple(n for n, p in fields_sig.parameters.items()
                     if p.kind in {p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD}
                     and p.default is p.empty)

        kw = {n: p.default for n, p in fields_sig.parameters.items()
              if p.kind in {p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY}
              and p.default is not p.empty}

        namespace["ARGS"] = args[1:]
        namespace["KWARGS"] = kw
        namespace["ALL_ARGS"] = tuple(fields_sig.parameters.keys())[1:]

        return namespace

    @classmethod
    def term_in_bases(cls, bases):
        """ Is `Transformer` in bases? """
        for base in bases:
            if base is Transformer or cls.term_in_bases(base.__bases__):
                return True

        return False

    def __new__(mcls, name, bases, namespace):
        # Check methods on any subclasses of Transformer
        # and expand the subclass namespace
        if mcls.term_in_bases(bases):
            namespace = mcls._expand_namespace(name, namespace)

        return super(TransformerMetaClass, mcls).__new__(
            mcls, name, bases, namespace)


class Transformer(metaclass=TransformerMetaClass):
    result_type = staticmethod(result_type)


class LMTransformer(Transformer):
    OUTPUTS = ["lm"]

    def fields(self, radec, phase_centre):
        if not isinstance(radec, types.Array) or radec.ndim != 2:
            raise ValueError(f"{radec} must be a (source, radec) array")

        if not isinstance(phase_centre, types.Array) or radec.ndim != 1:
            raise ValueError(f"{phase_centre} must be a 1D array")

        dt = self.result_type(radec.dtype, phase_centre.dtype)
        return [("lm", types.Array(dt, radec.ndim, radec.layout))]

    def transform(self, radec, phase_centre):
        pass

    def dask_schema(self, radec, phase_centre):
        return {"radec": ("source", "radec"),
                "phase_centre": ("radec",)}
