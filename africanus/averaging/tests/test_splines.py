# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from africanus.averaging.splines import (fit_cubic_spline,
                                         evaluate_spline)


# Generate y,z coords from given x coords
def generate_y_coords(x):
    y = -0.5 * x**2 - 0.3 * x + 5.0
    # z = 0.1 * x**3 + 5
    return y


@pytest.mark.flaky(min_passes=1, max_runs=3)
@pytest.mark.parametrize("order", [0])
def test_fit_cubic_spline(order):
    # Generate function y for x
    x = np.linspace(-2.0, 2.0, 16)
    y = generate_y_coords(x)

    spline = fit_cubic_spline(x, y)

    # Evaluation of spline at knot points is exact
    sy = evaluate_spline(spline, x, order=order)
    assert_almost_equal(sy, y)

    # Evaluate spline at points between knots is pretty inaccurate
    dx = np.diff(x) / 2
    dx += x[:-1]
    dy = generate_y_coords(dx)
    sdy = evaluate_spline(spline, dx, order=order)
    assert_almost_equal(sdy, dy, decimal=2)
