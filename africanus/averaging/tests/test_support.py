# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from africanus.averaging.support import (unique_baselines, unique_time,
                                         generate_metadata)


@pytest.fixture
def time():
    return np.asarray([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])  # noqa


@pytest.fixture
def ant1():
    return np.asarray([0,   0,   1,   0,   0,   1,   2,   0,   0,   1],  # noqa
                      dtype=np.int32)


@pytest.fixture
def ant2():
    return np.asarray([1,   2,   2,   0,   1,   2,   3,   0,   1,   2], # noqa
                      dtype=np.int32)


@pytest.fixture
def vis():
    def _vis(row, chan, fcorrs):
        return (np.arange(row*chan*fcorrs, dtype=np.float32) +
                np.arange(1, row*chan*fcorrs+1, dtype=np.float32)*1j)

    return _vis


def test_simplified_lookup(time, ant1, ant2):
    from africanus.averaging.support import better_lookup

    utime, time_inv, _ = unique_time(time)
    ubl, bl_inv, _ = unique_baselines(ant1, ant2)
    mask = np.full((ubl.shape[0], utime.shape[0]), -1, dtype=np.int32)

    mask[bl_inv, time_inv] = np.arange(time.size)

    row_map, time_avg = better_lookup(time, ant1, ant2, time_bin_size=2)

    # Now recalculate time_avg using the row_map
    time_avg_2 = np.zeros_like(time_avg)
    counts = np.zeros(time_avg.shape, dtype=np.uint32)

    for i, r in enumerate(row_map):
        time_avg_2[r] += time[i]
        counts[r] += 1

    # Normalise by counts
    time_avg_2 /= counts

    assert_array_equal(time_avg, time_avg_2)


def test_unique_time(time):
    time = np.flipud(time)

    utime, inv, counts = unique_time(time)
    assert_array_equal(utime, [1.0, 2.0, 3.0])
    assert_array_equal(utime[inv], time)
    assert_array_equal(counts, [3, 4, 3])


def test_unique_baselines(ant1, ant2):
    ant1 = np.flipud(ant1)
    ant2 = np.flipud(ant2)

    bl, inv, counts = unique_baselines(ant1, ant2)
    assert_array_equal(bl, [[0, 0], [0, 1], [0, 2], [1, 2], [2, 3]])
    assert_array_equal(bl[inv], np.stack([ant1, ant2], axis=1))
    assert_array_equal(counts, [2, 3, 1, 3, 1])


def test_lookups(time, ant1, ant2, vis):
    tup = generate_metadata(time, ant1, ant2)
    row_lookup, time_lookup, out_lookup, out_rows, tbins, sentinel = tup

    # Another way of generating the row_lookup
    utime, time_inv, _ = unique_time(time)
    ubl, bl_inv, _ = unique_baselines(ant1, ant2)
    expected_row_lookup = np.full((ubl.shape[0], utime.shape[0]), -1,
                                  dtype=np.int32)
    expected_row_lookup[bl_inv, time_inv] = np.arange(time.size)

    expected_out_lookup = [3, 7, 10,  0,  4,  8,  1, 11, 12,
                           2,  5,  9,  6, 13, 14]

    assert_array_equal(expected_row_lookup, row_lookup)
    assert_array_equal(expected_out_lookup, out_lookup)
    assert_array_equal(out_rows, np.count_nonzero(row_lookup != -1))
