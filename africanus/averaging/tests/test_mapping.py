# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from africanus.averaging.support import unique_time, unique_baselines
from africanus.averaging.row_mapping import row_mapper
from africanus.averaging.channel_mapping import channel_mapper


@pytest.fixture
def time():
    return np.asarray([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])  # noqa


@pytest.fixture
def ant1():
    return np.asarray([0,   0,   1,   0,   0,   1,   2,   0,   0,   1],  # noqa
                      dtype=np.int32)


@pytest.fixture
def ant2():
    return np.asarray([1,   2,   2,   0,   1,   2,   3,   0,   1,   2],  # noqa
                      dtype=np.int32)


def test_row_mapper(time, ant1, ant2):
    utime, time_inv, _ = unique_time(time)
    ubl, bl_inv, _ = unique_baselines(ant1, ant2)
    mask = np.full((ubl.shape[0], utime.shape[0]), -1, dtype=np.int32)

    mask[bl_inv, time_inv] = np.arange(time.size)

    row_map, time_avg = row_mapper(time, ant1, ant2, time_bin_size=2)

    # Now recalculate time_avg using the row_map
    time_avg_2 = np.zeros_like(time_avg)
    counts = np.zeros(time_avg.shape, dtype=np.uint32)

    # Add times at row_map indices to time_avg_2
    np.add.at(time_avg_2, row_map, time)
    # Add 1 at row_map indices to counts
    np.add.at(counts, row_map, 1)
    # Normalise
    time_avg_2 /= counts

    ant1_avg = np.empty(time_avg.shape, dtype=ant1.dtype)
    ant2_avg = np.empty(time_avg.shape, dtype=ant2.dtype)

    ant1_avg[row_map] = ant1
    ant2_avg[row_map] = ant2

    assert_array_equal(time_avg, time_avg_2)


def test_channel_mapper():
    chan_map = channel_mapper(64, 17)

    uchan, counts = np.unique(chan_map, return_counts=True)

    assert_array_equal(chan_map[0:17], 0)
    assert_array_equal(chan_map[17:34], 1)
    assert_array_equal(chan_map[34:51], 2)
    assert_array_equal(chan_map[51:64], 3)

    assert_array_equal(uchan, [0, 1, 2, 3])
    assert_array_equal(counts, [17, 17, 17, 13])
