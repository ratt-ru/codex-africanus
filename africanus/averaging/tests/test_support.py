# -*- coding: utf-8 -*-


import numpy as np
from numpy.testing import assert_array_equal
import pytest

from africanus.averaging.support import (unique_baselines, unique_time)


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


@pytest.fixture
def vis():
    def _vis(row, chan, fcorrs):
        return (np.arange(row*chan*fcorrs, dtype=np.float32) +
                np.arange(1, row*chan*fcorrs+1, dtype=np.float32)*1j)

    return _vis


def test_unique_time(time):
    # Reverse time to test that sort works
    time = np.flipud(time)

    utime, idx, inv, counts = unique_time(time)
    assert_array_equal(utime, [1.0, 2.0, 3.0])
    assert_array_equal(utime[inv], time)
    assert_array_equal(time[idx], utime)
    assert_array_equal(counts, [3, 4, 3])


def test_unique_baselines(ant1, ant2):
    # Reverse ant1, ant2 to test that sort works
    ant1 = np.flipud(ant1)
    ant2 = np.flipud(ant2)

    test_bl = np.stack([ant1, ant2], axis=1)

    bl, idx, inv, counts = unique_baselines(ant1, ant2)
    assert_array_equal(bl, [[0, 0], [0, 1], [0, 2], [1, 2], [2, 3]])
    assert_array_equal(bl[inv], test_bl)
    assert_array_equal(test_bl[idx], bl)
    assert_array_equal(counts, [2, 3, 1, 3, 1])
