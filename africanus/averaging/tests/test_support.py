# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from africanus.averaging.support import (unique_baselines, unique_time,
                                         generate_lookups)


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


def test_unique_time(time):
    utime, inv = unique_time(time)
    assert_array_equal(utime[inv], time)


def test_unique_baselines(ant1, ant2):
    bl, inv = unique_baselines(ant1, ant2)
    assert_array_equal(bl[inv], np.stack([ant1, ant2], axis=1))


def test_lookups(time, ant1, ant2, vis):
    row_lookup, out_lookup = generate_lookups(time, ant1, ant2)

    # We have 5 unique baselines and 3 unique times
    expected_row_lookup = np.asarray([[-1,  3,  7],
                                      [0,  4,  8],
                                      [1, -1, -1],
                                      [2,  5,  9],
                                      [-1,  6, -1]])

    expected_out_lookup = [3, 7, 10,  0,  4,  8,  1, 11, 12,
                           2,  5,  9,  6, 13, 14]
    assert_array_equal(expected_row_lookup, row_lookup)
    assert_array_equal(expected_out_lookup, out_lookup)
