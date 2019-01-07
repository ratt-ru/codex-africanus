# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from africanus.rime import feed_rotation as np_feed_rotation
from africanus.rime.cuda.feeds import feed_rotation as cp_feed_rotation


@pytest.mark.parametrize("shape", [(10, 7), (8,)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cupy_feed_rotation(shape, dtype):
    cp = pytest.importorskip('cupy')

    pa = np.random.random(shape).astype(dtype)

    cp_feed_rot = cp_feed_rotation(cp.asarray(pa))
    np_feed_rot = feed_rotation(pa)

    assert np.allclose(cp.asnumpy(cp_feed_rot), np_feed_rot)
