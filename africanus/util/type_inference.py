
# -*- coding: utf-8 -*-


try:
    from dask.array import Array as dask_array
except ImportError:
    dask_array = object()

import numba
import numpy as np


def _numpy_dtype(arg):
    if isinstance(arg, np.ndarray):
        return arg.dtype
    elif isinstance(arg, numba.types.npytypes.Array):
        return np.dtype(arg.dtype.name)
    elif isinstance(arg, dask_array):
        return arg.dtype
    else:
        raise ValueError("Unhandled type %s" % type(arg))


def infer_complex_dtype(*args):
    """ Infer complex datatype from arg inputs """
    return np.result_type(np.complex64, *(_numpy_dtype(a) for a in args))
