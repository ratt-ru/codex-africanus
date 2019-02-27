# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from africanus.averaging.support import generate_lookups
from africanus.averaging.row_averager import row_average


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
    return np.asarray([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])


def test_new_averager(time, ant1, ant2, uvw, interval):
    metadata = generate_lookups(time, ant1, ant2, 2)
    row_lookup, time_lookup, out_lookup, out_rows, tbins, sentinel = metadata

    tup = row_average(time, ant1, ant2, metadata,
                      uvw, time, interval, interval)
    (time_avg, ant1_avg, ant2_avg,
     uvw_avg, time_centroid_avg,
     interval_avg, exposure_avg) = tup

    assert_array_equal(time_avg, [1.0, 1.5, 1.5, 2.0, 2.5, 3.0, 3.0])
    assert_array_equal(ant1_avg, [0, 0, 1, 2, 0, 0, 1])
    assert_array_equal(ant2_avg, [2, 1, 2, 3, 0, 1, 2])
    assert_array_equal(uvw_avg, [[2.0,  2.0,   2.0],
                                 [3.0,  3.0,   3.0],
                                 [4.5,  4.5,   4.5],
                                 [7.0,  7.0,   7.0],
                                 [6.0,  6.0,   6.0],
                                 [9.0,  9.0,   9.0],
                                 [10.0, 10.0, 10.0]])
    assert_array_equal(time_centroid_avg, time_avg)
    assert_array_equal(interval_avg, [2.0, 4.0, 4.0, 2.0, 4.0, 2.0, 2.0])
    assert_array_equal(exposure_avg, [2.0, 4.0, 4.0, 2.0, 4.0, 2.0, 2.0])
