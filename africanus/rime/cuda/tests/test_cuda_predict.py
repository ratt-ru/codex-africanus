# -*- coding: utf-8 -*-


import numpy as np
import pytest

from africanus.rime.predict import predict_vis as np_predict_vis
from africanus.rime.cuda.predict import predict_vis
from africanus.rime.tests.test_predict import (corr_shape_parametrization,
                                               die_presence_parametrization,
                                               dde_presence_parametrization,
                                               chunk_parametrization,
                                               rc)


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

    ant1 = np.concatenate([np.random.randint(0, a, rows, dtype=np.int32)
                           for rows in chunks['rows']])

    ant2 = np.concatenate([np.random.randint(0, a, rows, dtype=np.int32)
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
