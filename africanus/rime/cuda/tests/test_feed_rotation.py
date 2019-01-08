# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

    cp_feed_rot = cp.asnumpy(cp_feed_rot)

    if not np.allclose(cp_feed_rot, np_feed_rot):
        d = np.invert(np.isclose(cp_feed_rot, np_feed_rot))

        for idx in zip(*np.nonzero(d)):
            print(idx, cp_feed_rot[idx], np_feed_rot[idx])

        # for idx in np.asarray(np.nonzero(d)).T:
        #     print(idx, cp_feed_rot[idx], np_feed_rot[idx])

    assert np.allclose(cp_feed_rot, np_feed_rot)
