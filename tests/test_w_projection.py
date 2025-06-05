#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def test_polyfit2d():
    from africanus.filters.wproj.spheroidal import polyfit2d, polyval2d

    npoints = 10
    x = np.random.random(npoints)
    y = np.random.random(npoints)
    z = x**2 + y**2 + 3 * x**3 + y + np.random.random(npoints)
    coeffs = polyfit2d(x, y, z, order=5)
    rz = polyval2d(x, y, coeffs)

    assert np.allclose(z, rz)
