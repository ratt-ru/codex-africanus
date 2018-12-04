# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

from africanus.constants import minus_two_pi_over_c
from africanus.util.cuda import (cuda_function, format_kernel, grids,
                                 memoize_kernel)
from africanus.util.requirements import requires_optional

try:
    import cupy as cp
    from cupy.core._scalar import get_typename as _get_typename
    from cupy.cuda.compiler import CompileException
except ImportError:
    pass

try:
    from jinja2 import Template
except ImportError:
    pass

log = logging.getLogger(__name__)

_PHASE_DELAY_TEMPLATE = """
// #include <cupy/complex.cuh>
#include <cupy/carray.cuh>
// #include <cupy/atomics.cuh>

#define blockdimx {{blockdimx}}
#define blockdimy {{blockdimy}}

#define minus_two_pi_over_c {{minus_two_pi_over_c}}

extern "C" __global__ void {{kernel_name}}(
    const CArray<{{lm_type}}, 2> lm,
    const CArray<{{uvw_type}}, 2> uvw,
    const CArray<{{freq_type}}, 1> frequency,
    CArray<{{out_type}}2, 3> complex_phase)
{
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int chan = blockIdx.x*blockDim.x + threadIdx.x;

    // Return if outside the grid
    if(row >= uvw.shape()[0] || chan >= frequency.shape()[0])
        { return; }

    // Reinterpret inputs as vector types
    const {{lm_type}}2 * lm_ptr = reinterpret_cast<const {{lm_type}}2 *>(
                                    &lm[0]);
    const {{uvw_type}}3 * uvw_ptr = reinterpret_cast<const {{uvw_type}}3 *>(
                                    &uvw[0]);
    {{out_type}}2 * complex_phase_ptr = reinterpret_cast<{{out_type}}2 *>(
                                    &complex_phase[0]);

    __shared__ struct {
        {{uvw_type}}3 uvw[blockdimy];
        {{freq_type}} frequency[blockdimx];
    } shared;

    // UVW coordinates vary along y dimension only
    if(threadIdx.x == 0)
        { shared.uvw[threadIdx.y] = uvw_ptr[row]; }

    // Frequencies vary along x dimension only
    if(threadIdx.y == 0)
        { shared.frequency[threadIdx.x] = frequency[chan]; }

    __syncthreads();

    for(int source = 0; source < lm.shape()[0]; ++source)
    {
        {{lm_type}}2 rlm = lm_ptr[source];
        {{lm_type}} n = {{sqrt_fn}}(1.0 - rlm.x*rlm.x - rlm.y*rlm.y) - 1.0;
        {{out_type}} real_phase = rlm.x*shared.uvw[threadIdx.y].x +
                                 rlm.y*shared.uvw[threadIdx.y].y +
                                 n*shared.uvw[threadIdx.y].z;

        real_phase = minus_two_pi_over_c *
                     real_phase *
                     shared.frequency[threadIdx.x];

        {{out_type}}2 cplx_phase;
        {{sincos_fn}}(real_phase, &cplx_phase.y, &cplx_phase.x);


        ptrdiff_t idx [] = {source, row, chan};
        complex_phase[idx] = cplx_phase;
    }
}
"""


def _key_fn(lm, uvw, frequency):
    return (lm.dtype, uvw.dtype, frequency.dtype)


@memoize_kernel(_key_fn)
def _generate_kernel(lm, uvw, frequency):
    # Floating point output type
    out_dtype = np.result_type(lm, uvw, frequency)

    # Block sizes
    blockdimx = 32 if frequency.dtype == np.float32 else 16
    blockdimy = 32 if uvw.dtype == np.float32 else 16
    block = (blockdimx, blockdimy, 1)

    # Create template
    render = Template(_PHASE_DELAY_TEMPLATE).render
    name = "phase_delay"

    code = render(kernel_name=name,
                  lm_type=_get_typename(lm.dtype),
                  uvw_type=_get_typename(uvw.dtype),
                  freq_type=_get_typename(frequency.dtype),
                  out_type=_get_typename(out_dtype),
                  sqrt_fn=cuda_function('sqrt', lm.dtype),
                  sincos_fn=cuda_function('sincos', out_dtype),
                  minus_two_pi_over_c=minus_two_pi_over_c,
                  blockdimx=blockdimx,
                  blockdimy=blockdimy).encode('utf-8')

    # Complex output type
    out_dtype = np.result_type(out_dtype, np.complex64)
    return cp.RawKernel(code, name), block, code, out_dtype


@requires_optional("cupy", "jinja2")
def phase_delay(lm, uvw, frequency):
    """
    Cupy implementation of the phase delay kernel.

    TODO(sjperkins). Fill in the documentation with the numba doc template
    """
    kernel, block, code, out_dtype = _generate_kernel(lm, uvw, frequency)
    grid = grids((frequency.shape[0], uvw.shape[0], 1), block)
    out = cp.empty(shape=(lm.shape[0], uvw.shape[0], frequency.shape[0]),
                   dtype=out_dtype)

    try:
        kernel(grid, block, (lm, uvw, frequency, out))
    except CompileException:
        log.exception(format_kernel(code))
        raise

    return out
