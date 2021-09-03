import inspect

import numba
from numba.core import types
from numba.experimental import structref
from numba.np.numpy_support import as_dtype
import numpy as np


@structref.register
class StateStructRef(types.StructRef):
    def preprocess_fields(self, fields):
        """ Disallow literal types in field definitions """
        return tuple((n, types.unliteral(t)) for n, t in fields)


def sigcheck_factory(expected_sig):
    def check_constructor_signature(self, fn):
        sig = inspect.signature(fn)
        if sig != expected_sig:
            raise ValueError(f"{fn.__name__}{sig} should be "
                             f"{fn.__name__}{expected_sig}")

    return check_constructor_signature


class TermMetaClass(type):
    REQUIRED = ("fields", "initialiser", "dask_schema")

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
                raise ValueError(f"{name}.fields{fields_sig} "
                                 f"should be "
                                 f"{name}.fields(self, ...)")

            if p.kind == p.VAR_POSITIONAL:
                raise NotImplementedError(f"*{n} in fields{fields_sig} "
                                          f"is not supported")

            if p.kind == p.VAR_KEYWORD:
                raise NotImplementedError(f"**{n} in fields{fields_sig} "
                                          f"is not supported")

        dask_schema_sig = inspect.signature(methods["dask_schema"])

        if dask_schema_sig != fields_sig:
            raise TypeError(f"{name}.dask_schema{dask_schema_sig} "
                            f"should be "
                            f"{name}.dask_schema{fields_sig}")

        field_params = list(fields_sig.parameters.values())
        expected_init_params = field_params.copy()
        Parameter = inspect.Parameter
        state_param = Parameter("state", Parameter.POSITIONAL_OR_KEYWORD)
        expected_init_params.insert(1, state_param)
        expected_init_sig = fields_sig.replace(parameters=expected_init_params)

        init_sig = inspect.signature(methods["initialiser"])

        if expected_init_sig != init_sig:
            raise ValueError(f"{name}.initialiser{init_sig} "
                             f"should be "
                             f"{name}.fields{expected_init_sig}")

        args = tuple(n for n, p in fields_sig.parameters.items()
                     if p.kind in {p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD}
                     and p.default is p.empty)

        kw = {n: p.default for n, p in fields_sig.parameters.items()
              if p.kind in {p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY}
              and p.default is not p.empty}

        expected_init_params.pop(0)
        expected_init_sig = fields_sig.replace(parameters=expected_init_params)
        validator = sigcheck_factory(expected_init_sig)

        namespace = namespace.copy()
        namespace["ARGS"] = args[1:]
        namespace["KWARGS"] = kw
        namespace["ALL_ARGS"] = tuple(fields_sig.parameters.keys())[1:]
        namespace["validate_constructor"] = validator

        return namespace

    def __new__(mcls, name, bases, namespace):
        if bases:
            namespace = mcls._expand_namespace(name, namespace)

        return super(TermMetaClass, mcls).__new__(mcls, name, bases, namespace)


class Term(metaclass=TermMetaClass):
    @staticmethod
    def result_type(*args):
        arg_types = []

        for arg in args:
            if isinstance(arg, types.Type):
                if isinstance(arg, types.Array):
                    arg_types.append(as_dtype(arg.dtype))
                else:
                    arg_types.append(as_dtype(arg))

            elif isinstance(arg, np.generic):
                arg_types.append(arg)
            else:
                raise TypeError(f"Unknown type {type(arg)} of argument {arg}")

        return numba.typeof(np.result_type(*arg_types)).dtype
