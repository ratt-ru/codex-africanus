# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import wraps

try:
    from dask.utils import SerializableLock as Lock
except ImportError:
    from threading import Lock

import numpy as np


cuda_fns = {
    np.dtype(np.float32): {
        'sqrt': 'sqrtf',
        'sincos': 'sincosf',
        'sincospi': 'sincospif',
    },
    np.dtype(np.float64): {
        'sqrt': 'sqrt',
        'sincos': 'sincos',
        'sincospi': 'sincospi',
    },
}


def grids(dims, blocks):
    """
    Determine the grid size, given space dimensions sizes and blocks

    Parameters
    ----------
    dims : tuple of ints
        `(x, y, z)` tuple

    Returns
    -------
    tuple
        `(x, y, z)` grid size tuple
    """
    if not len(dims) == 3:
        raise ValueError("dims must be an (x, y, z) tuple. "
                         "CUDA dimension ordering is inverted compared "
                         "to NumPy")

    if not len(blocks) == 3:
        raise ValueError("blocks must be an (x, y, z) tuple. "
                         "CUDA dimension ordering is inverted compared "
                         "to NumPy")

    return tuple((d + b - 1) // b for d, b in zip(dims, blocks))


def format_kernel(code):
    """
    Formats some code with line numbers

    Parameters
    ----------
    code : str
        Code

    Returns
    -------
    str
        Code prefixed with line numbers
    """
    lines = ['']
    lines.extend(["%-5d %s" % (i, l) for i, l
                  in enumerate(code.split('\n'), 1)])
    return '\n'.join(lines)


def cuda_function(function_name, dtype):
    try:
        type_map = cuda_fns[dtype]
    except KeyError:
        raise ValueError("No registered functions for type %s" % dtype)

    try:
        return type_map[function_name]
    except KeyError:
        raise ValueError("Unknown CUDA function %s" % function_name)


class memoize_kernel(object):
    """ Decorate the compilation of CUDA kernels """
    def __init__(self, key_fn):
        self._key_fn = key_fn
        self._lock = Lock()
        self._cache = {}

    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = self._key_fn(*args, **kwargs)

            with self._lock:
                try:
                    return self._cache[key]
                except KeyError:
                    self._cache[key] = entry = fn(*args, **kwargs)
                    return entry

        return wrapper
