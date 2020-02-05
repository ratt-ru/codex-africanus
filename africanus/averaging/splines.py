# -*- coding: utf-8 -*-

from collections import namedtuple

import numpy as np

from africanus.util.numba import njit

A, B, C = range(3)
Spline = namedtuple("Spline", "ma mb mc mx my")

@njit(nogil=True, cache=True)
def solve_trid_system(x, y, left_type=2, right_type=2,
                     left_value=0.0, right_value=0.0):
    """
    Solves a tridiagonal matrix

    https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    """

    diag = np.zeros((x.shape[0],3), dtype=x.dtype)

    v = np.zeros_like(y)

    n = x.shape[0]

    for i in range(1, n-1):
        diag[i, A] = (1.0 / 3.0) * (x[i] - x[i-1])
        diag[i, B] = (2.0 / 3.0) * (x[i+1] - x[i-1])
        diag[i, C] = (1.0 / 3.0) * (x[i+1] - x[i])
        v[i] = (y[i+1] - y[i])/(x[i+1] - x[i]) - (y[i] - y[i-1])/(x[i] - x[i-1])

    # Configure left end point
    if left_type == 2:
        diag[0, A] = 0.0
        diag[0, B] = 2.0
        diag[0, C] = left_value
    elif left_type == 1:
        diag[0, A] = 1.0 * (x[1] - x[0])
        diag[0, B] = 2.0 * (x[1] - x[0])
        v[0] = 3.0 * ((y[1] - y[0]) / (x[1] - x[0]) - left_value)
    else:
        raise ValueError("left_type not in (1, 2)")

    # Configure right endpoint
    if right_type == 2:
        diag[n-1, B] = 2.0
        diag[n-1, C] = 0.0
        v[n-1] = right_value
    elif left_type == 1:
        diag[n-1, B] = 2.0 * (x[n-1] - x[n-2])
        diag[n-1, C] = 1.0 * (x[n-1] - x[n-2])
        v[n-1] = 3.0 * (right_value - (y[n-1] - y[n-2]) / (x[n-1] - x[n-2]))
    else:
        raise ValueError("right_type not in (1, 2)")

    cp = np.zeros_like(x)
    vp = np.zeros_like(v)
    x = np.zeros_like(v)

    # c prime and v prime
    cp[0] = diag[0, C] / diag[0, B]
    vp[0] = v[0] / diag[0, B]

    for i in range(1, n):
        cp[i] = diag[i, C] / (diag[i, B] - diag[i, A] * cp[i-1])
        vp[i] = (v[i] - diag[i, A]*vp[i-1]) / (diag[i, B] - diag[i, A]*cp[i-1])

    x[n-1] = vp[n-1]

    for i in range(n - 1, -1, -1):
        x[i] = vp[i] - cp[i] * x[i+1]

    return x


@njit(nogil=True, cache=True)
def fit_cubic_spline(x, y, left_type=2, right_type=2,
                     left_value=0.0, right_value=0.0):
    b = solve_trid_system(x, y, left_type, right_type, left_value, right_value)
    a = np.empty_like(b)
    c = np.empty_like(b)

    n = x.shape[0]

    for i in range(n - 1):
        a[i] = (b[i+1] - b[i]) / (3*(x[i+1] - x[i]))
        c[i] = ((y[i+1] - y[i]) / (x[i+1] - x[i])
                 - 1.0/3.0*(2.0*b[i] + b[i+1])*(x[i+1] - x[i]))

    h = x[n-2] - x[n-1]
    a[n-1] = 0
    c[n-1] = 3.0*a[n-2]*h*h + 2.0*b[n-2]*h + c[n-2]

    return Spline(a, b, c, x, y)

@njit(nogil=True, cache=True)
def evaluate_cubic_spline(spline, x):
    ma, mb, mc, mx, my = spline

    force_linear_extrapolation = False
    mb0 = mb[0] if not force_linear_extrapolation else 0.0
    mc0 = mc[0]

    values = np.empty_like(x)

    for i, p in enumerate(x):
        j = max(np.searchsorted(mx, p), 0)

        h = x[i] - mx[j]

        if p < x[0]:
            values[i] = (mb0*h + mc0)*h + my[0]
        elif p > x[-1]:
            values[i] = (mb[-1]*h + mc[-1])*h + my[-1]
        else:
            values[i] = ((ma[i]*h + mb[i])*h + mc[i])*h + my[i]

    return values

