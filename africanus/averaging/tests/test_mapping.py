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
def interval():
    data = np.asarray([1.9, 2.0, 2.1, 1.85, 1.95, 2.0, 2.05, 2.1, 2.05, 1.9])
    return data*0.1


@pytest.fixture
def ant1():
    return np.asarray([0,   0,   1,   0,   0,   1,   2,   0,   0,   1],  # noqa
                      dtype=np.int32)


@pytest.fixture
def ant2():
    return np.asarray([1,   2,   2,   0,   1,   2,   3,   0,   1,   2],  # noqa
                      dtype=np.int32)


def flag_row_factory(nrows, flagged_rows):
    flag_row = np.zeros(nrows, dtype=np.uint8)

    if flagged_rows is not None:
        flag_row[flagged_rows] = 1

    return flag_row


@pytest.mark.parametrize("time_bin_secs", [0.1, 0.2, 1, 2, 4])
@pytest.mark.parametrize("flagged_rows", [None, [0, 1], [2, 4], range(10)])
def test_row_mapper(time, interval, ant1, ant2,
                    flagged_rows, time_bin_secs):
    utime, _, time_inv, _ = unique_time(time)
    ubl, _, bl_inv, _ = unique_baselines(ant1, ant2)
    mask = np.full((ubl.shape[0], utime.shape[0]), -1, dtype=np.int32)

    mask[bl_inv, time_inv] = np.arange(time.size)

    flag_row = flag_row_factory(time.size, flagged_rows)

    row_map, time_avg, exp_sum = row_mapper(time, interval, ant1, ant2,
                                            flag_row=flag_row,
                                            time_bin_secs=time_bin_secs)

    in_rows = row_map[0, :]
    out_rows = row_map[1, :]

    # Now recalculate time_avg using the row_map
    time_avg_2 = np.zeros_like(time_avg)
    counts = np.zeros(time_avg.shape, dtype=np.uint32)

    # Add times at row_map indices to time_avg_2
    np.add.at(time_avg_2, out_rows, time[in_rows])
    # Add 1 at row_map indices to counts
    np.add.at(counts, out_rows, 1)
    # Normalise
    time_avg_2 /= counts

    ant1_avg = np.empty(time_avg.shape, dtype=ant1.dtype)
    ant2_avg = np.empty(time_avg.shape, dtype=ant2.dtype)

    ant1_avg[out_rows] = ant1[in_rows]
    ant2_avg[out_rows] = ant2[in_rows]

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
