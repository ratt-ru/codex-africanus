# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from africanus.averaging.avg_factory import averaging_factory


def test_averaging_factory():
    row = 40
    chan = 16
    corr = 4

    time = np.arange(row, dtype=np.float64)
    vis = (np.random.random((row, chan, corr)) +
           np.random.random((row, chan, corr))*1j)

    time_avg = averaging_factory("TimeAverager", time.shape[0], 5, time.dtype)
    vis_avg = averaging_factory("VisAverager", row, chan, corr,
                                5, 5, vis.dtype)

    for r in range(row):
        time_avg.add(time[r])
        vis_avg.add(vis[r])

    print(time_avg.result)
