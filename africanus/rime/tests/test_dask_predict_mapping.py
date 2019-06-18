# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from operator import mul

from africanus.compatibility import reduce
from africanus.rime.dask import predict_vis

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest


def rf(*args, **kw):
    return np.random.random(*args, **kw)


def rc(*args, **kw):
    return rf(*args, **kw) + rf(*args, **kw)*1j


@pytest.mark.parametrize("streams", [1, 2, 3])
def test_dask_stream_coherency_reduction(streams):
    dask = pytest.importorskip('dask')
    da = pytest.importorskip('dask.array')

    src = (1, 1, 1, 1, 1, 1, 1, 1, 1)
    ants = (7,)
    row = (16, 16, 16)
    times = (7, 5, 6)
    chan = (8, 8)
    corr = (2, 2)

    nsrc = sum(src)
    ntime = sum(times)
    na = sum(ants)
    nrow = sum(row)
    nchan = sum(chan)

    time_index = [np.random.randint(t, size=r) for t, r in zip(times, row)]
    ant1 = [np.random.randint(na, size=r) for r in row]
    ant2 = [np.random.randint(na, size=r) for r in row]
    time_index = np.concatenate(time_index)
    ant1 = np.concatenate(ant1)
    ant2 = np.concatenate(ant2)

    dde1 = rc((nsrc, ntime, na, nchan) + corr)
    coh = rc((nsrc, nrow, nchan) + corr)

    da_time_index = da.from_array(time_index, chunks=row)
    da_ant1 = da.from_array(ant1, chunks=row)
    da_ant2 = da.from_array(ant2, chunks=row)
    da_dde1 = da.from_array(dde1, chunks=(src, times, ants, chan) + corr)
    da_coh = da.from_array(coh, chunks=(src, row, chan) + corr)

    stream_red = predict_vis(da_time_index, da_ant1, da_ant2,
                             dde1_jones=da_dde1, source_coh=da_coh,
                             dde2_jones=da_dde1, streams=streams)

    fan_red = predict_vis(da_time_index, da_ant1, da_ant2,
                          dde1_jones=da_dde1, source_coh=da_coh,
                          dde2_jones=da_dde1, streams=None)

    assert_array_almost_equal(stream_red.compute(), fan_red.compute())
