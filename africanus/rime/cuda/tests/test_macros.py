# -*- coding: utf-8 -*-


from os.path import join as pjoin

import numpy as np
import pytest

from africanus.util.code import format_code
from africanus.util.jinja2 import jinja_env


@pytest.mark.parametrize("ncorrs", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", [np.int32, np.float64, np.float32,
                                   np.complex64, np.complex128])
@pytest.mark.parametrize("nvis", [9, 10, 11, 32, 1025])
@pytest.mark.parametrize("debug", ["false"])
def test_cuda_inplace_warp_transpose(ncorrs, dtype, nvis, debug):
    cp = pytest.importorskip('cupy')

    path = pjoin("rime", "cuda", "tests", "test_warp_transpose.cu.j2")
    render = jinja_env.get_template(path).render

    dtypes = {
        np.float32: 'float',
        np.float64: 'double',
        np.int32: 'int',
        np.complex64: 'float2',
        np.complex128: 'double2',
    }

    code = render(type=dtypes[dtype], warp_size=32,
                  corrs=ncorrs, debug=debug).encode("utf-8")
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
