# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from operator import mul

import numpy as np
import pytest

from africanus.compatibility import reduce


@pytest.mark.parametrize("corrs", [(1,), (2,), (2, 2)])
def test_time_and_channel_averaging(corrs):
    from africanus.averaging import time_and_channel

    time = np.asarray([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])  # noqa
    ant1 = np.asarray([0,   0,   1,   0,   0,   1,   2,   0,   0,   1])    # noqa
    ant2 = np.asarray([1,   2,   2,   0,   1,   2,   3,   0,   1,   2])    # noqa

    row = time.shape[0]
    chan = 17
    fcorrs = reduce(mul, corrs, 1)

    vis = (np.arange(row*chan*fcorrs, dtype=np.float32) +
           np.arange(1, row*chan*fcorrs+1, dtype=np.float32)*1j)

    vis = vis.reshape((row, chan) + corrs)

    avg_vis, avg_time, avg_ant1, avg_ant2 = time_and_channel(
                                time, ant1, ant2, vis,
                                time_bins=2, chan_bins=4,
                                return_time=True, return_antenna=True)

    np.testing.assert_array_almost_equal(avg_time,
                                         [1.0, 1.5, 1.5, 2.0, 2.5, 3.0, 3.0])

    np.testing.assert_array_almost_equal(avg_ant1, [0, 0, 1, 2, 0, 0, 1])
    np.testing.assert_array_almost_equal(avg_ant2, [2, 1, 2, 3, 0, 1, 2])

    # Same correlation shape
    assert vis.shape[2:] == avg_vis.shape[2:] == corrs

    assert vis.sum() == avg_vis.sum()
