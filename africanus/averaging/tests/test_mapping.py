# -*- coding: utf-8 -*-


import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest

from africanus.averaging.support import unique_time, unique_baselines
from africanus.averaging.time_and_channel_mapping import (row_mapper,
                                                          channel_mapper)


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

    ret = row_mapper(time, interval, ant1, ant2,
                     flag_row=flag_row,
                     time_bin_secs=time_bin_secs)

    # For TIME AND INTERVAL, flagged inputs can
    # contribute to unflagged outputs
    new_time = np.zeros_like(ret.time)
    new_interval = np.zeros_like(ret.interval)
    counts = np.zeros(ret.time.shape, dtype=np.uint32)
    np.add.at(new_time, ret.map, time)
    np.add.at(new_interval, ret.map, interval)
    np.add.at(counts, ret.map, 1)

    assert_array_equal(ret.time, new_time / counts)
    assert_array_equal(ret.interval, new_interval)

    # For TIME_CENTROID and EXPOSURE,
    # unflagged inputs only contribute to unflagged outputs and
    # flagged inputs only contribute to flagged outputs

    # Now recalculate time_avg using the row_map
    new_tc = np.zeros_like(ret.time)
    new_exp = np.zeros_like(ret.interval)
    counts = np.zeros(ret.time.shape, dtype=np.uint32)

    sel = flag_row == ret.flag_row[ret.map]
    np.add.at(new_tc, ret.map[sel], time[sel])
    np.add.at(new_exp, ret.map[sel], interval[sel])
    np.add.at(counts, ret.map[sel], 1)

    ant1_avg = np.empty(ret.time.shape, dtype=ant1.dtype)
    ant2_avg = np.empty(ret.time.shape, dtype=ant2.dtype)
    ant1_avg[ret.map[sel]] = ant1[sel]
    ant2_avg[ret.map[sel]] = ant2[sel]

    # Do it a different way
    new_tc2 = np.zeros_like(ret.time)
    new_exp2 = np.zeros_like(ret.interval)
    counts2 = np.zeros(ret.time.shape, dtype=np.uint32)

    for ri, ro in enumerate(ret.map):
        if flag_row[ri] == 1 and ret.flag_row[ro] == 1:
            new_tc2[ro] += time[ri]
            new_exp2[ro] += interval[ri]
            counts2[ro] += 1
        elif flag_row[ri] == 0 and ret.flag_row[ro] == 0:
            new_tc2[ro] += time[ri]
            new_exp2[ro] += interval[ri]
            counts2[ro] += 1

    assert_array_almost_equal(new_tc / counts, new_tc2 / counts2)
    assert_array_almost_equal(new_exp, new_exp2)


def test_channel_mapper():
    chan_map, out_chans = channel_mapper(64, 17)

    uchan, counts = np.unique(chan_map, return_counts=True)

    assert_array_equal(chan_map[0:17], 0)
    assert_array_equal(chan_map[17:34], 1)
    assert_array_equal(chan_map[34:51], 2)
    assert_array_equal(chan_map[51:64], 3)

    assert_array_equal(uchan, [0, 1, 2, 3])
    assert_array_equal(counts, [17, 17, 17, 13])

    assert out_chans == 4
