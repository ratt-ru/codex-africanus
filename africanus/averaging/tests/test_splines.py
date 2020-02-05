# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from africanus.averaging.splines import fit_cubic_spline, evaluate_cubic_spline


@pytest.fixture
def x():
    return np.array([1.0, 2.3, 3.5, 4.0, 4.2, 5.6])

@pytest.fixture
def y():
    return np.array([1.1, 1.5, 1.8, 2.2, 3.7, 9.5])

def test_fit_cubic_spline(x, y):
    spline = fit_cubic_spline(x, y)
    ny = evaluate_cubic_spline(spline, x)
    assert_almost_equal(y, ny)

    
