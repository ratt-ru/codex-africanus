# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from africanus.util.cupy import format_kernel


def test_shuffle():
    cp = pytest.importorskip("cupy")
    jinja2 = pytest.importorskip("jinja2")

    _TEMPLATE = jinja2.Template("""
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

        __syncthreads();

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


def test_shuffle_2():
    cp = pytest.importorskip("cupy")
    jinja2 = pytest.importorskip("jinja2")

    _TEMPLATE = jinja2.Template("""
    #include <cupy/carray.cuh>
    // #include <trove/transpose.h>

    #define warp_size 32
    //#define base_idx(lane_id) (lane_id / {{corrs}})
    // #define corr_idx(lane_id) (lane_id % {{corrs}})

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

        __syncthreads();

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

            printf("lane %d target_corr %d src_lane %d "
                   "src_corr %d value %d\\n",
                   lane_id, corr, src_lane, in_corr,
                   __shfl_sync(mask, loads[in_corr],
                               src_lane, warp_size));


            values[corr] = __shfl_sync(mask, loads[in_corr],
                                         src_lane, warp_size);
        }

        __syncthreads();

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

    import pkg_resources
    from os.path import join as pjoin

    include_path = pkg_resources.resource_filename("africanus",
                                                   pjoin("include", "trove"))

    code = _TEMPLATE.render(type=dtypes[dtype], corrs=ncorrs).encode("utf-8")
    print(format_kernel(code))
    kernel = cp.RawKernel(code, "kernel", options=("-I %s" % include_path,))

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


def test_shuffle_3():
    cp = pytest.importorskip("cupy")
    jinja2 = pytest.importorskip("jinja2")

    _TEMPLATE = jinja2.Template("""
    #include <cupy/carray.cuh>

    #define warp_size 32
    #define base_idx(lane_id) (lane_id & (warp_size - {{corrs}}))
    #define corr_idx(lane_id) (lane_id & ({{corrs}} - 1))

    extern "C" __global__ void kernel(
        const int * input,
        int * output,
        int N)
    {
        int n = blockIdx.x*blockDim.x + threadIdx.x;
        int lane_id = threadIdx.x & (warp_size - 1);

        if(n >= N)
            { return; }

        // Input correlation handled by this thread
        int in_corr = corr_idx(lane_id);
        int mask = __activemask();

        if(threadIdx.x == 0)
            { printf("mask %d\\n", mask); }

        {{type}} loads[{{corrs}}];
        {{type}} values[{{corrs}}];

        {% for corr in range(corrs) %}
        loads[{{corr}}] = input[n + {{corr}}*N];
        {%- endfor %}

        __syncthreads();

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

            printf("lane %d target_corr %d src_lane %d "
                   "src_corr %d value %d\\n",
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
        output[n + {{corr}}*N] = values[{{corr}}];
        {%- endfor %}
    }
    """)

    N = 32
    corrs = 4
    dtype = np.int32

    dtypes = {
        np.float32: 'float',
        np.float64: 'double',
        np.int32: 'int',
    }

    code = _TEMPLATE.render(type=dtypes[dtype], corrs=corrs).encode("utf-8")
    print(format_kernel(code))
    kernel = cp.RawKernel(code, "kernel")

    inputs = cp.arange(N*corrs, dtype=dtype).reshape(N, corrs)
    outputs = cp.empty_like(inputs)
    args = (inputs, outputs, N)
    block = (256, 1, 1)
    grid = tuple((d + b - 1) // b for d, b in zip((N, 1, 1), block))

    print(grid, block)
    kernel(grid, block, args)

    print("blah\n")
    print("\n")
    print(inputs)
    print(outputs)

