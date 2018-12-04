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
    return code
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
    """
    Memoize the compilation of CUDA kernels. Slightly more advanced
    version of a standard python memoization as it takes a key function
    which should return a custom key for caching the kernel,
    based on the arguments passed to the kernel.

    In the following example, the arguments required to generate
    the `phase_delay` kernel are the types of `lm`, `uvw`
    and `frequency`, as well as the number of correlations `ncorr`.

    They ``key_fn`` produces a unique key based on these types
    and the number of correlations:

    .. code-block:: python

        def key_fn(lm, uvw, frequency, ncorrs=4):
            ''' Produce a unique key for the arguments of _generate_phase_delay_kernel '''
            return (lm.dtype, uvw.dtype, frequency.dtype, ncorrs)

        _code_template = jinja2.Template('''
        #define ncorrs {{ncorrs}}

        __global__ void phase_delay(
            const {{lm_type}} * lm,
            const {{uvw_type}} * uvw,
            const {{freq_type}} * frequency,
            {{out_type}} * out)
        {
            ...
        }
        ''')

        _type_map = {
            np.float32: 'float',
            np.float64: 'double'
        }

        @memoize_kernel(key_fn)
        def _generate_phase_delay_kernel(lm, uvw, frequency, ncorrs=4):
            ''' Generate the phase delay kernel '''
            out_dtype = np.result_type(lm.dtype, uvw.dtype, frequency.dtype)
            code = _code_template.render(lm_type=_type_map[lm.dtype],
                                         uvw_type=_type_map[uvw.dtype],
                                         freq_type=_type_map[frequency.dtype],
                                         ncorrs=ncorrs)
            return cp.RawKernel(code, "phase_delay")
    """
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
