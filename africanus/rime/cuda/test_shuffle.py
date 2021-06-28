# -*- coding: utf-8 -*-


import numpy as np
import pytest

from africanus.util.code import format_code


def test_cuda_shuffle_transpose():
    cp = pytest.importorskip("cupy")
    jinja2 = pytest.importorskip("jinja2")

    _TEMPLATE = jinja2.Template("""
    #include <cupy/carray.cuh>

    #define debug {{debug}}

    extern "C" __global__ void kernel(
        const CArray<{{type}}, 2> input,
        CArray<{{type}}, 2> output)
    {
        const ptrdiff_t & nvis = input.shape()[0];
        int v = blockIdx.x*blockDim.x + threadIdx.x;
        int lane_id = threadIdx.x & ({{warp_size}} - 1);

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
                                     src_lane, {{warp_size}});
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
                                     src_lane, {{warp_size}});
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

    code = _TEMPLATE.render(type=dtypes[dtype], warp_size=32,
                            corrs=ncorrs, debug="false")
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


def register_assign_cycles(N, case=0):
    """
    Determine cycles that stem from performing
    an in-place assignment of the elements of an array.
    In the following. we cannot naively assign the source index
    to the dest index,
    If we assigned the value at index 0 to index 1, the assignment
    from index 1 to index 3 would be invalid, for example:

    src:  [3, 0, 2, 1]
    dest: [0, 1, 2, 3]

    assignment cycles can be broken by using a
    temporary variable to store the contents of a source index
    until dependent assignments have been performed.

    In this way, the minimum number of registers can be used
    to perform the in-place assignment.

    Returns
    -------
    list of lists of tuples
        For example, `[[(0, 2), (2, 0)], [1, 3], [3, 1]]`

    """
    dest = range(N)
    src = [(N - case + n) % N for n in dest]

    deps = {d: s for d, s in zip(dest, src) if d != s}

    for di, d in enumerate(dest):
        si = src.index(d)
        if si > di:
            deps[si] = di

    cycles = []

    while len(deps) > 0:
        k, v = deps.popitem()
        cycle = [(k, v)]

        while True:
            try:
                k = v
                v = deps.pop(k)
            except KeyError:
                # Check that the last key we're trying
                # to get is the first one in the cycle
                assert k == cycle[0][0]
                break

            cycle.append((k, v))

        cycles.append(cycle)

    return cycles


class CupyTemplatingException(Exception):
    def __init__(self, msg):
        super(CupyTemplatingException, self).__init__(msg)


def throw_helper(msg):
    raise CupyTemplatingException(msg)


@pytest.mark.parametrize("ncorrs", [1, 2, 4, 8])
def test_cuda_shuffle_transpose_2(ncorrs):
    cp = pytest.importorskip("cupy")
    jinja2 = pytest.importorskip("jinja2")

    # Implement a warp transpose using Kepler's register shuffle instructions
    # as described in del Mundo's
    # `Towards a performance-portable FFT library for heterogeneous computing`
    # https://doi.org/10.1145/2597917.2597943
    # https://homes.cs.washington.edu/~cdel/papers/cf14-fft.pdf
    # The poster is especially informative
    # https://homes.cs.washington.edu/~cdel/posters/073113-on-efficacy-shuffle-sc2013.pdf
    # and
    # `Enabling Efficient Intra-Warp Communication for
    #  Fourier Transforms in a Many-Core Architecture.`
    # https://homes.cs.washington.edu/~cdel/papers/sc13-shuffle-abstract.pdf
    # http://sc13.supercomputing.org/sites/default/files/PostersArchive/spost142.html

    _TEMPLATE = jinja2.Template("""
    #include <cupy/carray.cuh>

    {%- if (corrs < 1 or (corrs.__and__(corrs - 1) != 0)) %}
    {{ throw("corrs must be 1 or a power of 2") }}
    {%- endif %}

    {% macro warp_transpose(var_name, var_type, var_length, tmp_name="tmp") %}
    {% if var_length > 1 %}
        {
            int mask = __activemask();
            int case_id = threadIdx.x & {{var_length - 1}};
            {{var_type}} {{tmp_name}};  // For variable swaps

            // Horizontal (inter-thread) Rotation
            int addr = case_id;
            {%- for case in range(var_length) %}
            {{var_name}}[{{case}}] = __shfl_sync(mask, {{var_name}}[{{case}}], addr, {{var_length}});
            {%- if not loop.last %}
            addr = __shfl_sync(mask, addr, (case_id + 1) & {{var_length - 1}}, {{var_length}});
            {%- endif %}
            {%- endfor %}

            // Vertical (intra-thread) Rotation
            {%- for case in range(var_length) %}
            // Case {{case}}
            {%- set cycles = register_assign_cycles(corrs, case) %}
            {%- for cycle in cycles %}
            {%- set cstart = cycle[0][0] %}
            {{tmp_name}} = {{var_name}}[{{cstart}}];
            {%- for dest, src in cycle %}
            {%- set src_var = tmp_name if cstart == src else var_name + "[" + src|string + "]" %}
            {{var_name}}[{{dest}}] = case_id == {{case}} ? {{src_var}} : {{var_name}}[{{dest}}];
            {%- endfor %}
            {%- endfor %}
            {%- endfor %}

            // Horizontal (inter-thread) Rotation
            addr = ({{var_length}} - case_id) & {{var_length - 1}};
            {%- for case in range(var_length) %}
            {{var_name}}[{{case}}] = __shfl_sync(mask, {{var_name}}[{{case}}], addr, {{var_length}});
            {%- if not loop.last %}
            addr = __shfl_sync(mask, addr, (case_id + {{var_length - 1}}) & {{var_length - 1}}, {{var_length}});
            {%- endif %}
            {%- endfor %}
        }
    {%- endif %}
    {%- endmacro %}

    {%- set width = corrs %}

    extern "C" __global__ void kernel(
        const CArray<{{type}}, 2> input,
        CArray<{{type}}, 2> output)
    {
        const ptrdiff_t & nvis = input.shape()[0];
        int v = blockIdx.x*blockDim.x + threadIdx.x;

        if(v >= nvis)
            { return; }

        // Array to hold our variables
        {{type}} values[{{corrs}}];

        {% for corr in range(corrs) %}
        values[{{corr}}] = input[v + {{corr}}*nvis];
        {%- endfor %}

        if({{debug}})
        {
            if(threadIdx.x == 0)
                { printf("mask %d\\n", __activemask()); }

            printf("[%d] %d %d %d %d\\n",
                   threadIdx.x & {{warp_size - 1}},
                   values[0], values[1],
                   values[2], values[3]);

            if(threadIdx.x == 0)
                { printf("\\n"); }
        }

        {{ warp_transpose("values", type, corrs) }}
        {{ warp_transpose("values", type, corrs) }}

        if({{debug}})
        {
            if(threadIdx.x == 0)
                { printf("\\n"); }

            printf("[%d] %d %d %d %d\\n",
                   threadIdx.x & {{warp_size - 1}},
                   values[0], values[1],
                   values[2], values[3]);
        }

        {% for corr in range(corrs) %}
        output[v + {{corr}}*nvis] = values[{{corr}}];
        {%- endfor %}
    }
    """)  # noqa

    nvis = 32
    dtype = np.int32

    dtypes = {
        np.float32: 'float',
        np.float64: 'double',
        np.int32: 'int',
    }

    code = _TEMPLATE.render(type=dtypes[dtype],
                            throw=throw_helper,
                            register_assign_cycles=register_assign_cycles,
                            warp_size=32,
                            corrs=ncorrs,
                            debug="false")
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
