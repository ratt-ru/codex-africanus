# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


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
    return tuple((d + b - 1) // b for d, b in zip(dims, blocks))


def format_kernel(code):
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
