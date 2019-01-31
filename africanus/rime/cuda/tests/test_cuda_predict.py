# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from africanus.rime.predict import predict_vis as np_predict_vis
from africanus.rime.cuda.predict import predict_vis
from africanus.rime.tests.test_predict import (corr_shape_parametrization,
                                               die_presence_parametrization,
                                               dde_presence_parametrization,
                                               chunk_parametrization,
                                               rf, rc)


@corr_shape_parametrization
@dde_presence_parametrization
@die_presence_parametrization
@chunk_parametrization
def test_cuda_predict_vis(corr_shape, idm, einsum_sig1, einsum_sig2,
                          a1j, blj, a2j, g1j, bvis, g2j,
                          chunks):
    np.random.seed(40)

    cp = pytest.importorskip('cupy')

    s = sum(chunks['source'])
    t = sum(chunks['time'])
    a = sum(chunks['antenna'])
    c = sum(chunks['channels'])
    r = sum(chunks['rows'])

    a1_jones = rc((s, t, a, c) + corr_shape)
    bl_jones = rc((s, r, c) + corr_shape)
    a2_jones = rc((s, t, a, c) + corr_shape)
    g1_jones = rc((t, a, c) + corr_shape)
    base_vis = rc((r, c) + corr_shape)
    g2_jones = rc((t, a, c) + corr_shape)

    # Add 10 to the index to test time index normalisation
    time_idx = np.concatenate([np.full(rows, i+10, dtype=np.int32)
                               for i, rows in enumerate(chunks['rows'])])

    ant1 = np.concatenate([np.random.randint(0, a, rows)
                           for rows in chunks['rows']])

    ant2 = np.concatenate([np.random.randint(0, a, rows)
                           for rows in chunks['rows']])

    assert ant1.size == r

    model_vis = predict_vis(cp.asarray(time_idx),
                            cp.asarray(ant1),
                            cp.asarray(ant2),
                            cp.asarray(a1_jones) if a1j else None,
                            cp.asarray(bl_jones) if blj else None,
                            cp.asarray(a2_jones) if a2j else None,
                            cp.asarray(g1_jones) if g1j else None,
                            cp.asarray(base_vis) if bvis else None,
                            cp.asarray(g2_jones) if g2j else None)

    np_model_vis = np_predict_vis(time_idx,
                                  ant1,
                                  ant2,
                                  a1_jones if a1j else None,
                                  bl_jones if blj else None,
                                  a2_jones if a2j else None,
                                  g1_jones if g1j else None,
                                  base_vis if bvis else None,
                                  g2_jones if g2j else None)

    np.testing.assert_array_almost_equal(cp.asnumpy(model_vis), np_model_vis)

    assert model_vis.shape == (r, c) + corr_shape


@pytest.mark.skip
def test_cuda_xor_shuffle_min_reduce():
    cp = pytest.importorskip('cupy')
    from africanus.util.jinja2 import jinja_env
    from africanus.util.cuda import grids

    source = r"""
    #include <cupy/carray.cuh>

    extern "C"
    __global__ void reduce(CArray<int, 2> data, CArray<int, 1> output)
    {
        int col = blockIdx.y*blockDim.y + threadIdx.y;
        int row = blockIdx.x*blockDim.x + threadIdx.x;

        bool pred = col >= data.shape()[1] || row >= data.shape()[0];

        if(pred)
            { return; }


        unsigned int mask = __ballot_sync(0xFFFFFFF, !pred);

        int bid = blockIdx.y*gridDim.x + blockIdx.x;
        int tid = threadIdx.y*{{blockdimx}} + threadIdx.x;
        int lane = tid % {{warp_size}};
        int wid = tid / {{warp_size}};

        int value = data[col*data.shape()[0] + row];
        int minval = value;

        #pragma unroll
        for(int i = {{warp_size // 2}}; i > 0; i /= 2)
        {
            int v = __shfl_xor_sync(mask, minval, i);
            bool dest_valid = ((1 << i) & mask) > 0;
            printf("%d: %d %d\n", i, dest_valid, v);


            if(dest_valid)
            {
                minval = min(minval, __shfl_down_sync(mask, minval, i));
            }
        }

        printf("block %d warp %d mask %d: %d %d\n",
               bid, wid, mask, value, minval);

        output[0] = minval;

    }

    """

    render = jinja_env.from_string(source).render
    block = (16, 16, 1)
    code = render(blockdimx=block[0], blockdimy=block[1], warp_size=32)
    code = code.encode('utf-8')

    kernel = cp.RawKernel(code, "reduce")
    row, col = 7, 5
    grid = grids((row, col, 1), block)
    data = cp.arange(row*col, dtype=cp.int32).reshape(col, row) + 1
    output = cp.empty((1,), dtype=cp.int32)

    kernel(grid, block, (data, output))
