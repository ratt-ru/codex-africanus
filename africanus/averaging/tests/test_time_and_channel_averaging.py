# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest


def test_time_and_channel_averaging():
    from africanus.averaging import time_and_channel

    time = np.asarray([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])  # noqa
    ant1 = np.asarray([  0,   0,   1,   0,   0,   1,   2,   0,   0,   1])  # noqa
    ant2 = np.asarray([  1,   2,   2,   0,   1,   2,   3,   0,   1,   2])  # noqa

    row = time.shape[0]
    chan = 17
    corr = 4

    vis = (np.arange(row*chan*corr, dtype=np.float32) +
           np.arange(1, row*chan*corr+1, dtype=np.float32)*1j)

    vis = vis.reshape((row, chan, corr))
    avg_vis = time_and_channel(time, ant1, ant2, vis, time_bins=2, chan_bins=4)

    assert vis.sum() == avg_vis.sum()
