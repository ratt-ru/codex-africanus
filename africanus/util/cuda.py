# -*- coding: utf-8 -*-


import numpy as np

_array_types = [np.ndarray]

try:
    import dask.array as da
except ImportError:
    pass
else:
    _array_types.append(da.Array)

try:
    import cupy as cp
except ImportError:
    pass
else:
    _array_types.append(cp.ndarray)

_array_types = tuple(_array_types)

cuda_fns = {
    np.dtype(np.float32): {
        'abs': 'fabsf',
        'cos': 'cosf',
        'floor': 'floorf',
        'make2': 'make_float2',
        'max': 'fmaxf',
        'min': 'fminf',
        'rsqrt': 'rsqrtf',
        'sqrt': 'sqrtf',
        'sin': 'sinf',
        'sincos': 'sincosf',
        'sincospi': 'sincospif',
    },
    np.dtype(np.float64): {
        'abs': 'fabs',
        'cos': 'cos',
        'floor': 'floor',
        'make2': 'make_double2',
        'max': 'fmax',
        'min': 'fmin',
        'rsqrt': 'rsqrt',
        'sin': 'sin',
        'sincos': 'sincos',
        'sincospi': 'sincospi',
        'sqrt': 'sqrt',
    },
}


numpy_to_cuda_type_map = {
    np.dtype('int8'): "char",
    np.dtype('uint8'): "unsigned char",
    np.dtype('int16'): "short",
    np.dtype('uint16'): "unsigned short",
    np.dtype('int32'): "int",
    np.dtype('uint32'): "unsigned int",
    np.dtype('float32'): "float",
    np.dtype('float64'): "double",
    np.dtype('complex64'): "float2",
    np.dtype('complex128'): "double2"
}

# Also map the types
numpy_to_cuda_type_map.update({k.type: v
                               for k, v
                               in numpy_to_cuda_type_map.items()})


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


def cuda_function(function_name, dtype):
    try:
        type_map = cuda_fns[dtype]
    except KeyError:
        raise ValueError("No registered functions for type %s" % dtype)

    try:
        return type_map[function_name]
    except KeyError:
        raise ValueError("Unknown CUDA function %s" % function_name)


def cuda_type(dtype):
    if isinstance(dtype, _array_types):
        dtype = dtype.dtype

    try:
        return numpy_to_cuda_type_map[dtype]
    except KeyError:
        raise ValueError("No registered map for type %s" % dtype)
