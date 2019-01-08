# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

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

    #  Row indices into the above time/ant indexed arrays
    time_idx = np.asarray([0, 0, 1, 1, 2, 2, 2, 2, 3, 3])
    ant1 = np.asarray([0, 0, 0, 0, 1, 1, 1, 2, 2, 3])
    ant2 = np.asarray([0, 1, 2, 3, 1, 2, 3, 2, 3, 3])

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

    assert model_vis.shape == (r, c) + corr_shape

