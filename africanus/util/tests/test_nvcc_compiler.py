from __future__ import print_function

import pytest


def test_nvcc_compiler(tmpdir):
    from pkg_resources import resource_filename
    from os.path import join as pjoin

    from africanus.util.nvcc import compile_using_nvcc

    cp = pytest.importorskip('cupy')

    cub_path = resource_filename("africanus", pjoin("include", "cub"))

    code = """
    #include <cupy/carray.cuh>
    #include <cub/cub.cuh>

    extern "C" __global__ void kernel(const CArray<int, 1> in,
                                      CArray<int, 1> out)
    {
        int x = blockDim.x*blockIdx.x + threadIdx.x;
        int n = int(in.shape()[0]);
        int value = in[x];
        // printf("[%d, %d] value = %d\\n", x, n, value);
        out[x] = value;
    }

    """
    mod = compile_using_nvcc(code, options=['-I ' + cub_path])
    kernel = mod.get_function("kernel")
    inputs = cp.arange(1024, dtype=cp.int32)
    outputs = cp.empty_like(inputs)

    kernel((1, 1, 1), (1024, 1, 1), (inputs, outputs))
    cp.testing.assert_array_almost_equal(inputs, outputs)
