# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from os.path import join as pjoin
from operator import mul

import numpy as np

from africanus.compatibility import reduce
from africanus.util.jinja2 import jinja_env
from africanus.rime.zernike import _ZERNIKE_DOCSTRING
from africanus.util.code import memoize_on_key, format_code
from africanus.util.cuda import cuda_function, cuda_type, grids
from africanus.util.requirements import requires_optional

try:
    import cupy as cp
    from cupy.cuda.compiler import CompileException
except ImportError:
    pass

log = logging.getLogger(__name__)


def _key_fn(coords, coeffs, noll_index):
    return (coords.dtype, coords.ndim,
            coeffs.dtype, coeffs.ndim,
            coeffs.shape[2],  # Memoize on number of polynomials
            coeffs.shape[3:], # Memoize on correlations
            noll_index.dtype, noll_index.ndim)


_TEMPLATE_PATH = pjoin("rime", "cuda", "zernike.cu.j2")


@memoize_on_key(_key_fn)
def _generate_kernel(coords, coeffs, noll_index):
    # Floating point output type
    out_dtype = np.result_type(coords, coeffs, noll_index, np.complex64)

    npoly = coeffs.shape[2]
    corr_shape = coeffs.shape[3:]
    ncorrs = reduce(mul, corr_shape, 1)

    # Block sizes
    blockdimx = 32
    blockdimy = 4
    blockdimz = 1
    block = (blockdimx, blockdimy, blockdimz)

    # Create template
    render = jinja_env.get_template(_TEMPLATE_PATH).render
    name = "zernike_dde"

    code = render(kernel_name=name,
                  coords_type=cuda_type(coords.dtype),
                  coords_dims=coords.ndim,
                  coeffs_type=cuda_type(coeffs.dtype),
                  coeffs_dims=coeffs.ndim,
                  noll_index_type=cuda_type(noll_index.dtype),
                  noll_index_dims=noll_index.ndim,
                  npoly=npoly,
                  corrs=ncorrs,
                  sin_fn=cuda_function('cos', coords.dtype),
                  cos_fn=cuda_function('sin', coords.dtype),
                  pow_fn=cuda_function('pow', coords.dtype),
                  sqrt_fn=cuda_function('sqrt', coords.dtype),
                  atan2_fn=cuda_function('atan2', coords.dtype),
                  out_type=cuda_type(out_dtype),
                  out_dims=coords.ndim + len(coeffs.shape[2:-1]),
                  blockdimx=blockdimx,
                  blockdimy=blockdimy,
                  blockdimz=blockdimz).encode('utf-8')

    # Complex output type
    out_dtype = np.result_type(out_dtype, np.complex64)
    return cp.RawKernel(code, name), block, out_dtype


@requires_optional("cupy")
def zernike_dde(coords, coeffs, noll_index):
    kernel, block, out_dtype = _generate_kernel(coords, coeffs, noll_index)
    sources, times, ants, chans = coords.shape[1:]
    corrs = coeffs.shape[2:-1]
    fcorrs = reduce(mul, corrs, 1)

    grid = grids((ants, chans, 1), block)
    out = cp.empty((sources, times, ants, chans, fcorrs), coeffs.dtype)

    try:
        kernel(grid, block, (coords, coeffs, noll_index, out))
    except CompileException:
        log.exception(format_code(kernel.code))
        raise

    return out


try:
    zernike_dde.__doc__ = _ZERNIKE_DOCSTRING.substitute(
                                array_type=':class:`cupy.ndarray`')
except AttributeError:
    pass
