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
    const CArray<{{type}}2, 2> input,
    CArray<{{type}}2, 2> output)
{
    int v = blockIdx.x*blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x & (warp_size - 1);

    if(v >= input.shape()[0])
        { return; }

    // Input correlation handled by this thread
    int in_corr = corr_idx(lane_id);
    int mask = __activemask();

    if(threadIdx.x == 0)
        { printf("mask %d\\n", mask); }

    {{type}}2 values[{{corrs}}];
    {{type}}2 loads[{{corrs}}];

    {% for corr in range(corrs) %}
    loads[{{corr}}] = input[v + {{corr}}*input.shape()[0]];
    {%- endfor %}

    printf("[%d, %d] %f %f %f %f\\n",
           lane_id, base_idx(lane_id),
           loads[0].x, loads[1].x,
           loads[2].x, loads[3].x);

    if(threadIdx.x == 0)
        { printf("\\n"); }

    // #pragma unroll ({{corrs}})
    for(int corr=0; corr < {{corrs}}; ++corr)
    {
        int src_lane = base_idx(lane_id) + corr;
        // int src_lane = threadIdx.x;
        // int src_lane = lane_id + corr;

        printf("thread %d src_lane %d corr %d \\n",
               threadIdx.x,  src_lane, in_corr);


        values[corr].x = __shfl_sync(mask, loads[in_corr].x,
                                     src_lane, warp_size);
        values[corr].y = __shfl_sync(mask, loads[in_corr].y,
                                     src_lane, warp_size);
    }

    if(threadIdx.x == 0)
        { printf("\\n"); }

    __syncthreads();

    printf("[%d, %d] %f %f %f %f\\n",
           lane_id, base_idx(lane_id),
           values[0].x, values[1].x,
           values[2].x, values[3].x);


    {% for corr in range(corrs) %}
    output[v + {{corr}}*input.shape()[0]] = values[{{corr}}];
    {%- endfor %}
}
""")

nvis = 10
ncorrs = 4


code = _TEMPLATE.render(type="double", corrs=ncorrs).encode("utf-8")
print(format_kernel(code))
kernel = cp.RawKernel(code, "kernel")


inputs = cp.arange(nvis*ncorrs, dtype=np.complex128).reshape(nvis, ncorrs)
outputs = cp.empty_like(inputs)
args = (inputs, outputs)
block = (256, 1, 1)
grid = tuple((d + b - 1) // b for d, b in zip((nvis, 1, 1), block))

print(grid, block)
kernel(grid, block, args)
print(inputs)
print(outputs)
