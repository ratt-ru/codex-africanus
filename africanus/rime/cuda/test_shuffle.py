# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from africanus.util.code import format_code


def test_cuda_shuffle_transpose():
    cp = pytest.importorskip("cupy")
    jinja2 = pytest.importorskip("jinja2")

    _TEMPLATE = jinja2.Template("""
    #include <cupy/carray.cuh>

    #define warp_size 32
    #define debug {{debug}}

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
        int mask = __activemask();


        {{type}} loads[{{corrs}}];
        {{type}} values[{{corrs}}];

        {% for corr in range(corrs) %}
        loads[{{corr}}] = input[v + {{corr}}*nvis];
        {%- endfor %}

        __syncthreads();

        if(debug)
        {
            if(threadIdx.x == 0)
                { printf("mask %d\\n", mask); }

            printf("[%d] %d %d %d %d\\n",
                   lane_id,
                   loads[0], loads[1],
                   loads[2], loads[3]);

            if(threadIdx.x == 0)
                { printf("\\n"); }
        }



        // Tranpose forward
        #pragma unroll ({{corrs}})
        for(int corr=0; corr < {{corrs}}; ++corr)
        {
            int src_corr = ({{corrs}} - corr + lane_id) % {{corrs}};
            int dest_corr = (lane_id + corr) % {{corrs}};
            int src_lane = (lane_id / {{corrs}})*{{corrs}} + dest_corr;

            values[dest_corr] = __shfl_sync(mask, loads[src_corr],
                                     src_lane, warp_size);
        }

        // Copy
        #pragma unroll ({{corrs}})
        for(int corr=0; corr < {{corrs}}; ++corr)
        {
            loads[corr] = values[corr];
        }

        // Transpose backward
        #pragma unroll ({{corrs}})
        for(int corr=0; corr < {{corrs}}; ++corr)
        {
            int src_corr = ({{corrs}} - corr + lane_id) % {{corrs}};
            int dest_corr = (lane_id + corr) % {{corrs}};
            int src_lane = (lane_id / {{corrs}})*{{corrs}} + dest_corr;

            values[dest_corr] = __shfl_sync(mask, loads[src_corr],
                                     src_lane, warp_size);
        }


        __syncthreads();

        if(debug)
        {
            if(threadIdx.x == 0)
                { printf("\\n"); }

            printf("[%d] %d %d %d %d\\n",
                   lane_id,
                   values[0], values[1],
                   values[2], values[3]);
        }


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

    code = _TEMPLATE.render(type=dtypes[dtype], corrs=ncorrs,
                            debug="true").encode("utf-8")
    kernel = cp.RawKernel(code, "kernel")

    inputs = cp.arange(nvis*ncorrs, dtype=dtype).reshape(nvis, ncorrs)
    outputs = cp.empty_like(inputs)
    args = (inputs, outputs)
    block = (256, 1, 1)
    grid = tuple((d + b - 1) // b for d, b in zip((nvis, 1, 1), block))

    try:
        kernel(grid, block, args)
    except cp.cuda.compiler.CompileException:
        print(format_code(kernel.code))
        raise

    np.testing.assert_array_almost_equal(cp.asnumpy(inputs),
                                         cp.asnumpy(outputs))
    return

    # Dead code
    print(grid, block)
    print("\n")
    print(inputs)
    print(outputs)
