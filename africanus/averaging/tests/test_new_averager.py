# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from africanus.averaging.support import unique_time, unique_baselines
from africanus.averaging.row_mapping import row_mapper
from africanus.averaging.new_averager import row_average


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


def _gen_testing_lookup(time, ant1, ant2, flag_row, time_bin_size):
    """
    Generates the same lookup as row_mapper, but different.

    Returns
    -------
    list of (float, (int, int), list of lists)
        Each tuple in the list corresponds to an output row, and
        is composed of `(avg_time, (ant1, ant2), binned_input_rows)`

    """
    utime, time_inv, _ = unique_time(time)
    ubl, bl_inv, _ = unique_baselines(ant1, ant2)
    bl_time_lookup = np.full((ubl.shape[0], utime.shape[0]), -1,
                             dtype=np.int32)

    # Create the row index
    row_idx = np.arange(time.size)

    # Set flagged rows to -1 to indicate missing data
    row_idx[flag_row > 0] = -1

    # Assign the row indices
    bl_time_lookup[bl_inv, time_inv] = row_idx

    # Create the time, baseline, row map
    time_bl_row_map = []

    for bl, (a1, a2) in enumerate(ubl):
        bl_row_idx = tmp = bl_time_lookup[bl, :]
        # Removing missing rows
        bl_row_idx = bl_row_idx[bl_row_idx != -1]
        # Split the row indices on the size of the time bin
        split_idx = np.arange(time_bin_size, bl_row_idx.size, time_bin_size)
        bin_map = [bm for bm in np.split(bl_row_idx, split_idx)
                   if bm.size != 0]

        # Produce a (avg_time, bl, rows) tuple
        time_bl_row_map.extend((time[rows].mean(), (a1, a2), rows.tolist())
                               for rows in bin_map)

    # Sort lookup sorted on averaged times
    return sorted(time_bl_row_map, key=lambda tup: tup[0])


@pytest.mark.parametrize("flagged_rows", [
    [], [8, 9], [4], [0, 1],
])
@pytest.mark.parametrize("time_bin_size", [1, 2, 3, 4])
def test_row_averager(time, ant1, ant2, flagged_rows,
                      uvw, interval, weight, sigma,
                      vis, flag, time_bin_size):

    flag_row = np.zeros(time.shape, dtype=np.uint8)
    flag_row[flagged_rows] = 1

    time_bl_row_map = _gen_testing_lookup(time, ant1, ant2,
                                          flag_row, time_bin_size)

    metadata = row_mapper(time, ant1, ant2, flag_row, time_bin_size)
    row_lookup, time_avg = metadata

    # Check that the averaged times from the test and accelerated lookup match
    assert_array_equal([t for t, _, _ in time_bl_row_map], time_avg)

    exposure = interval

    tup = row_average(metadata, ant1, ant2,
                      uvw, time, interval, interval,
                      weight, sigma)
    (time_avg, ant1_avg, ant2_avg,
     uvw_avg, time_centroid_avg,
     interval_avg, exposure_avg,
     weight_avg, sigma_avg) = tup

    # Input rows associated with each output row
    row_idx = [row for _, _, row in time_bl_row_map]

    # Take mean time, but first ant1 and ant2
    expected_times = [time[i].mean(axis=0) for i in row_idx]
    expected_ant1 = [ant1[i[0]] for i in row_idx]
    expected_ant2 = [ant2[i[0]] for i in row_idx]

    # Take mean average, but sum of interval and exposure
    expected_uvw = [uvw[i].mean(axis=0) for i in row_idx]
    expected_interval = [interval[i].sum(axis=0) for i in row_idx]
    expected_exposure = [exposure[i].sum(axis=0) for i in row_idx]
    expected_weight = [weight[i].mean(axis=0) for i in row_idx]
    expected_sigma = [sigma[i].mean(axis=0) for i in row_idx]

    assert_array_equal(time_avg, expected_times)
    assert_array_equal(ant1_avg, expected_ant1)
    assert_array_equal(ant2_avg, expected_ant2)
    assert_array_equal(time_centroid_avg, time_avg)
    assert_array_equal(uvw_avg, expected_uvw)
    assert_array_equal(interval_avg, expected_interval)
    assert_array_equal(exposure_avg, expected_exposure)
    assert_array_equal(weight_avg, expected_weight)
    assert_array_equal(sigma_avg, expected_sigma)
