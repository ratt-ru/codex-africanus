import numba
from numba.core import types
from numba.np.numpy_support import as_dtype
import numpy as np


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
