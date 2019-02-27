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


def test_new_averager(time, ant1, ant2):
    metadata = generate_lookups(time, ant1, ant2, 2)
    row_lookup, time_lookup, out_lookup, out_rows, tbins, sentinel = metadata

    # print("row_lookup", row_lookup)
    # print("time_lookup", time_lookup)
    # print("out_lookup", out_lookup)

    time_avg, ant1_avg, ant2_avg = row_average(time, ant1, ant2, metadata)

    np.testing.assert_array_almost_equal(time_avg,
                                         [1.0, 1.5, 1.5, 2.0, 2.5, 3.0, 3.0])

    np.testing.assert_array_almost_equal(ant1_avg, [0, 0, 1, 2, 0, 0, 1])
    np.testing.assert_array_almost_equal(ant2_avg, [2, 1, 2, 3, 0, 1, 2])

