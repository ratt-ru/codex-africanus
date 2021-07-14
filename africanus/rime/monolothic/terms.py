import numba
from numba.core import types
from numba.np.numpy_support import as_dtype
import numpy as np
import abc


class TermStructRef(types.StructRef):
    def preprocess_fields(self, fields):
        """ Disallow literal types in field definitions """
        return tuple((n, types.unliteral(t)) for n, t in fields)


class Term(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def term_type(cls, *args):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def initialiser(cls, *args):
        raise NotImplementedError

    @classmethod
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
