# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from africanus.averaging.support import generate_metadata
from africanus.averaging.row_averager import row_average, row_chan_average


@pytest.fixture
def time():
    return np.asarray([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])  # noqa


@pytest.fixture
def ant1():
    return np.asarray([0,   0,   1,   0,   0,   1,   2,   0,   0,   1],
                      dtype=np.int32)


@pytest.fixture
def ant2():
    return np.asarray([1,   2,   2,   0,   1,   2,   3,   0,   1,   2],
                      dtype=np.int32)


@pytest.fixture
def uvw():
    return np.asarray([[1.0,   1.0,  1.0],
                       [2.0,   2.0,  2.0],
                       [3.0,   3.0,  3.0],
                       [4.0,   4.0,  4.0],
                       [5.0,   5.0,  5.0],
                       [6.0,   6.0,  6.0],
                       [7.0,   7.0,  7.0],
                       [8.0,   8.0,  8.0],
                       [9.0,   9.0,  9.0],
                       [10.0, 10.0, 10.0]])


@pytest.fixture
def interval():
    return np.asarray([1.9, 2.0, 2.1, 1.85, 1.95, 2.0, 2.05, 2.1, 2.05, 1.9])


@pytest.fixture
def weight():
    shape = (10, 4)
    return np.arange(np.product(shape), dtype=np.float64).reshape(shape)


@pytest.fixture
def sigma():
    shape = (10, 4)
    return np.arange(np.product(shape), dtype=np.float64).reshape(shape)


@pytest.fixture
def vis():
    def _vis(row, chan, fcorrs):
        return (np.arange(row*chan*fcorrs, dtype=np.float32) +
                np.arange(1, row*chan*fcorrs+1, dtype=np.float32)*1j)

    return _vis


@pytest.fixture
def flag():
    def _flag(row, chan, fcorrs):
        return np.random.randint(0, 2, (row, chan, fcorrs))

    return _flag


def test_row_averager(time, ant1, ant2, uvw, interval, weight, sigma,
                      vis, flag):
    metadata = generate_metadata(time, ant1, ant2, 2)
    row_lookup, time_lookup, out_lookup, out_rows, tbins, sentinel = metadata

    exposure = interval

    tup = row_average(metadata, time, ant1, ant2,
                      uvw, time, interval, interval,
                      weight, sigma)
    (time_avg, ant1_avg, ant2_avg,
     uvw_avg, time_centroid_avg,
     interval_avg, exposure_avg,
     weight_avg, sigma_avg) = tup

    # Write out expected times, ant1 and ant2
    # Baseline 0 [0-2]: 1.0   -- from baselines [1]
    # Baseline 1 [0-1]: 1.5   -- from baselines [0, 4]
    # Baseline 2 [1-2]: 1.5   -- from baselines [2, 5]
    # Baseline 3 [2-3]: 2.0   -- from baselines [6]
    # Baseline 4 [0-0]: 2.5   -- from baselines [3, 7]
    # Baseline 5 [0-1]: 3.0   -- from baselines [8]
    # Baseline 6 [1-2]: 3.0   -- from baselines [9]
    written_ant1 = np.array([0, 0, 1, 2, 0, 0, 1])
    written_ant2 = np.array([2, 1, 2, 3, 0, 1, 2])
    written_times = np.array([1.0,  1.5,  1.5,  2.0,  2.5, 3.0,  3.0])

    idx = [[1], [0, 4], [2, 5], [6], [3, 7], [8], [9]]
    bin_size = np.array([len(i) for i in idx])

    # Take mean time, but first ant1 and ant2
    expected_times = [time[i].sum() for i in idx] / bin_size
    expected_ant1 = [ant1[i[0]] for i in idx]
    expected_ant2 = [ant2[i[0]] for i in idx]

    assert_array_equal(written_times, expected_times)
    assert_array_equal(written_ant1, expected_ant1)
    assert_array_equal(written_ant2, expected_ant2)

    # Take mean average, but sum of interval and exposure
    expected_uvw = [uvw[i].sum(axis=0) for i in idx] / bin_size[:, None]
    expected_interval = [interval[i].sum() for i in idx]
    expected_exposure = [exposure[i].sum() for i in idx]
    expected_weight = [weight[i].sum(axis=0) for i in idx] / bin_size[:, None]
    expected_sigma = [sigma[i].sum(axis=0) for i in idx] / bin_size[:, None]

    assert_array_equal(time_avg, expected_times)
    assert_array_equal(ant1_avg, expected_ant1)
    assert_array_equal(ant2_avg, expected_ant2)
    assert_array_equal(uvw_avg, expected_uvw)
    assert_array_equal(time_centroid_avg, time_avg)
    assert_array_equal(interval_avg, expected_interval)
    assert_array_equal(exposure_avg, expected_exposure)
    assert_array_equal(weight_avg, expected_weight)
    assert_array_equal(sigma_avg, expected_sigma)

    row_chan_average(metadata,
                     vis(10, 64, 4).reshape(10, 64, 4),
                     flag(10, 64, 4).reshape(10, 64, 4),
                     chan_bin_size=16)
