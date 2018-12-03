# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from africanus.util.cupy import format_kernel

import cupy as cp
import numpy as np
from jinja2 import Template


_TEMPLATE = Template("""
#include <cupy/carray.cuh>

#define warp_size 32
#define base_idx(lane_id) (lane_id & (warp_size - {{corrs}}))
#define corr_idx(lane_id) (lane_id & ({{corrs}} - 1))

extern "C" __global__ void kernel(
    const CArray<{{type}}, 2> input,
    CArray<{{type}}, 2> output)
{
    const ptrdiff_t & nvis = input.shape()[0];
    int v = blockIdx.x*blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x & (warp_size - 1);

    if(v >= nvis)
        { return; }

    // Input correlation handled by this thread
    int in_corr = corr_idx(lane_id);
    int mask = __activemask();

    if(threadIdx.x == 0)
        { printf("mask %d\\n", mask); }

    {{type}} loads[{{corrs}}];
    {{type}} values[{{corrs}}];

    {% for corr in range(corrs) %}
    loads[{{corr}}] = input[v + {{corr}}*nvis];
    {%- endfor %}

    printf("[%d, %d] %d %d %d %d\\n",
           lane_id, base_idx(lane_id),
           loads[0], loads[1],
           loads[2], loads[3]);

    if(threadIdx.x == 0)
        { printf("\\n"); }


    // #pragma unroll ({{corrs}})
    for(int corr=0; corr < {{corrs}}; ++corr)
    {
        int src_lane = base_idx(lane_id) + corr;
        // int src_lane = threadIdx.x;
        // int src_lane = lane_id + corr;

        printf("lane %d target_corr %d src_lane %d src_corr %d value %d\\n",
               lane_id, corr, src_lane, in_corr,
               __shfl_sync(mask, loads[in_corr],
                           src_lane, warp_size));


        values[corr] = __shfl_sync(mask, loads[in_corr],
                                     src_lane, warp_size);
    }

    if(threadIdx.x == 0)
        { printf("\\n"); }

    printf("[%d, %d] %d %d %d %d\\n",
           lane_id, base_idx(lane_id),
           values[0], values[1],
           values[2], values[3]);


    {% for corr in range(corrs) %}
    output[v + {{corr}}*nvis] = values[{{corr}}];
    {%- endfor %}
}
""")

nvis = 32
ncorrs = 4
dtype = np.int32

dtypes = {
    np.float32: 'float',
    np.float64: 'double',
    np.int32: 'int',
}


code = _TEMPLATE.render(type=dtypes[dtype], corrs=ncorrs).encode("utf-8")
print(format_kernel(code))
kernel = cp.RawKernel(code, "kernel")


inputs = cp.arange(nvis*ncorrs, dtype=dtype).reshape(nvis, ncorrs)
outputs = cp.empty_like(inputs)
args = (inputs, outputs)
block = (256, 1, 1)
grid = tuple((d + b - 1) // b for d, b in zip((nvis, 1, 1), block))

print(grid, block)
kernel(grid, block, args)


print("blah\n")
print("\n")
print(inputs)
print(outputs)