# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from africanus.gridding.facet.support import duvw_dtime


def test_duvw_dtime():
    ntime = 10

    ant1, ant2 = (a.astype(np.int32) for a in np.triu_indices(7, 1))
    nbl = ant1.size

    times = np.linspace(1.0, 10.0, ntime)

    time = np.repeat(times, nbl)
    ant1 = np.tile(ant1, ntime)
    ant2 = np.tile(ant2, ntime)

    uvw = np.random.random((ntime*nbl, 3))

    duvw_dt = duvw_dtime(time, ant1, ant2, uvw)
    assert duvw_dt.shape == uvw.shape
