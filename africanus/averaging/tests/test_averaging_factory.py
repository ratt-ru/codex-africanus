# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from africanus.averaging.time_avg_factory import time_averager_factory
from africanus.averaging.row_iterator import row_iterator


def test_row_iterator():
    it = row_iterator(7, 3)
    assert next(it) == (0, 0, False)
    assert next(it) == (1, 0, False)
    assert next(it) == (2, 0, True)
    assert next(it) == (3, 1, False)
    assert next(it) == (4, 1, False)
    assert next(it) == (5, 1, True)
    assert next(it) == (6, 2, True)

    with pytest.raises(StopIteration):
        next(it)

    it = row_iterator(6, 3)
    assert next(it) == (0, 0, False)
    assert next(it) == (1, 0, False)
    assert next(it) == (2, 0, True)
    assert next(it) == (3, 1, False)
    assert next(it) == (4, 1, False)
    assert next(it) == (5, 1, True)

    with pytest.raises(StopIteration):
        next(it)

    it = row_iterator(5, 3)
    assert next(it) == (0, 0, False)
    assert next(it) == (1, 0, False)
    assert next(it) == (2, 0, True)
    assert next(it) == (3, 1, False)
    assert next(it) == (4, 1, True)

    with pytest.raises(StopIteration):
        next(it)

    it = row_iterator(5, 2)
    assert next(it) == (0, 0, False)
    assert next(it) == (1, 0, True)
    assert next(it) == (2, 1, False)
    assert next(it) == (3, 1, True)
    assert next(it) == (4, 2, True)

    with pytest.raises(StopIteration):
        next(it)


def test_averaging_factory():
    row = 40

    time = np.arange(row, dtype=np.float64)

    TimeAverager = time_averager_factory(time.dtype)
    time_avg = TimeAverager(time.shape[0], 5, time.dtype)

    for r in range(row):
        time_avg.add(time[r])

    print(time_avg.result)
