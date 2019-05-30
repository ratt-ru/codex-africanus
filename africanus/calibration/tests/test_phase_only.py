# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.testing import assert_array_equal
import pytest

n_time = 32
n_freq = 64
n_dir = 3
n_ant = 7

@pytest.fixture
def give_times():
    return np.linspace(0,1, n_time)

@pytest.fixture
def give_freqs():
    return np.linspace(1e9,1e9, n_freq)

@pytest.fixture
def give_lm():
    l = np.random.randn(n_dir)
    m = np.random.randn(n_dir)
    ll, mm = np.meshgrid(l, m)
    lm = np.vstack((ll.flatten, mm.flatten)).T
    return l, m, lm


