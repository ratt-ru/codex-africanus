import numba
from numba.core import types
from numba.np.numpy_support import as_dtype
import numpy as np
import abc

class SignatureAdapter:
    __slots__ = ("signature",)

    def __init__(self, signature):
        # We don't support *args or **kwargs
        for n, p in signature.parameters.items():
            if p.kind == p.VAR_POSITIONAL:
                raise NotImplementedError(f"*{n} is not supported")
            elif p.kind == p.VAR_KEYWORD:
                raise NotImplementedError(f"**{n} is not supported")

        self.signature = signature

    def __eq__(self, other):
        return (type(other) is SignatureAdapter and
                self.signature == other.signature)

    @property
    def args(self):
        return tuple(n for n, p in self.signature.parameters.items()
                     if p.kind in {p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD}
                     and p.default is p.empty)

    @property
    def kwargs(self):
        return {n: p.default for n, p in self.signature.parameters.items()
                if p.kind in {p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY}
                and p.default is not p.empty}

class TermStructRef(types.StructRef):
    def preprocess_fields(self, fields):
        """ Disallow literal types in field definitions """
        return tuple((n, types.unliteral(t)) for n, t in fields)


class Term(abc.ABC):
    @abc.abstractmethod
    def term_type(cls, *args):
        raise NotImplementedError

    @abc.abstractmethod
    def initialiser(cls, *args):
        raise NotImplementedError

    @abc.abstractmethod
    def sampler(cls):
        raise NotImplementedError

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