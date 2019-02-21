# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from operator import mul

import numpy as np
import pytest

from africanus.compatibility import reduce


@pytest.fixture
def time():
    return np.asarray([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])  # noqa


@pytest.fixture
def ant1():
    return np.asarray([0,   0,   1,   0,   0,   1,   2,   0,   0,   1])    # noqa


@pytest.fixture
def ant2():
    return np.asarray([1,   2,   2,   0,   1,   2,   3,   0,   1,   2])    # noqa


@pytest.fixture
def vis():
    def _vis(row, chan, fcorrs):
        return (np.arange(row*chan*fcorrs, dtype=np.float32) +
                np.arange(1, row*chan*fcorrs+1, dtype=np.float32)*1j)

    return _vis


@pytest.mark.parametrize("corrs", [(1,), (2,), (2, 2)])
def test_time_and_channel_averaging(time, ant1, ant2, vis, corrs):
    from africanus.averaging import time_and_channel

    # time = np.asarray([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])  # noqa
    # ant1 = np.asarray([0,   0,   1,   0,   0,   1,   2,   0,   0,   1])    # noqa
    # ant2 = np.asarray([1,   2,   2,   0,   1,   2,   3,   0,   1,   2])    # noqa

    row = time.shape[0]
    chan = 5
    fcorrs = reduce(mul, corrs, 1)

    vis = vis(row, chan, fcorrs).reshape((row, chan) + corrs)
    flags = np.zeros(vis.shape, dtype=np.uint8)

    # Test no averaging case
    avg_vis, avg_time, avg_ant1, avg_ant2 = time_and_channel(
                                time, ant1, ant2, vis, flags,
                                avg_time=None, avg_chan=None,
                                return_time=True, return_antenna=True)

    np.testing.assert_array_almost_equal(avg_ant1, ant1)
    np.testing.assert_array_almost_equal(avg_ant2, ant2)
    np.testing.assert_array_almost_equal(avg_time, time)
    np.testing.assert_array_almost_equal(avg_vis, vis)

    # Now do some averaging
    avg_vis, avg_time, avg_ant1, avg_ant2 = time_and_channel(
                                time, ant1, ant2, vis, flags,
                                avg_time=2, avg_chan=2,
                                return_time=True, return_antenna=True)

    np.testing.assert_array_almost_equal(avg_time,
                                         [1.0, 1.5, 1.5, 2.0, 2.5, 3.0, 3.0])

    np.testing.assert_array_almost_equal(avg_ant1, [0, 0, 1, 2, 0, 0, 1])
    np.testing.assert_array_almost_equal(avg_ant2, [2, 1, 2, 3, 0, 1, 2])

    # Same correlation shape
    assert vis.shape[2:] == avg_vis.shape[2:] == corrs

    # This works if we comment out both time and channel
    # bin normalisation in time_and_channel
    # assert vis.sum() == avg_vis.sum()


@pytest.mark.parametrize("corrs", [(1,), (2,), (2, 2)])
def test_dask_time_and_channel_averaging(time, ant1, ant2, vis, corrs):
    """
    This doesn't test much and especially doesn't not test that the
    numpy version exactly matches that produced by the dask version.
    This is because the dask version does not average across chunk
    boundaries.
    """
    da = pytest.importorskip('dask.array')

    from africanus.averaging.dask import time_and_channel

    dask_time = da.concatenate([time]*3)
    dask_ant1 = da.concatenate([ant1]*3)
    dask_ant2 = da.concatenate([ant2]*3)

    rc = dask_time.chunks[0]
    fc = (5, 5)

    row = sum(rc)
    chan = sum(fc)
    fcorrs = reduce(mul, corrs, 1)
    avg_time = 2
    avg_chan = 2

    vis = vis(row, chan, fcorrs).reshape((row, chan) + corrs)
    dask_vis = da.from_array(vis, chunks=(rc, fc) + corrs)
    dask_flags = da.zeros(dask_vis.shape, chunks=(rc, fc) + corrs,
                          dtype=np.uint8)

    avg_vis = time_and_channel(dask_time, dask_ant1, dask_ant2,
                               dask_vis, dask_flags,
                               avg_time=avg_time, avg_chan=avg_chan,
                               return_time=False, return_antenna=False)

    avg_vis, avg_time, avg_ant1, avg_ant2 = time_and_channel(
                                dask_time, dask_ant1, dask_ant2,
                                dask_vis, dask_flags,
                                avg_time=avg_time, avg_chan=avg_chan,
                                return_time=True, return_antenna=True)

    expected_chans = sum((c + avg_chan - 1) // avg_chan
                         for c in dask_vis.chunks[1])

    avg_vis = avg_vis.compute()
    avg_time = avg_time.compute()
    avg_ant1 = avg_ant1.compute()
    avg_ant2 = avg_ant2.compute()

    assert avg_vis.shape[1] == expected_chans
