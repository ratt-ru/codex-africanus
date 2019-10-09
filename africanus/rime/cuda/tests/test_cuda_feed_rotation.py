# -*- coding: utf-8 -*-


import numpy as np
import pytest

from africanus.rime import feed_rotation as np_feed_rotation
from africanus.rime.cuda.feeds import feed_rotation as cp_feed_rotation


@pytest.mark.parametrize("feed_type", ["linear", "circular"])
@pytest.mark.parametrize("shape", [(10, 7), (8,)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cuda_feed_rotation(feed_type, shape, dtype):
    cp = pytest.importorskip('cupy')

    pa = np.random.random(shape).astype(dtype)

    cp_feed_rot = cp_feed_rotation(cp.asarray(pa), feed_type=feed_type)
    np_feed_rot = np_feed_rotation(pa, feed_type=feed_type)

    np.testing.assert_array_almost_equal(cp.asnumpy(cp_feed_rot), np_feed_rot)
