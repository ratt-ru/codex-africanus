#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

import pytest

def test_brightness():
    import numpy as np
    from africanus.stokes import brightness

    # One stokes
    stokes = np.asarray([[1], [5]])
    B = brightness(stokes, polarisation_type='linear')
    assert np.all(B == [[1],[5]])

    B = brightness(stokes, polarisation_type='circular')
    assert np.all(B == [[1],[5]])

    # Two stokes parameters
    stokes = np.asarray([[1, 2], [5, 6]], dtype=np.float64)

    # Linear (I and Q)
    B = brightness(stokes, polarisation_type='linear')
    expected = np.asarray([[1+2, 1-2], [5+6, 5-6]])
    assert np.all(B == expected)

    # Circular (I and V) produce the same result as linear
    B = brightness(stokes, polarisation_type='circular')
    assert np.all(B == expected)

    # Four stokes parameters
    stokes = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float64)

    # Linear (I,Q,U,V)
    B = brightness(stokes, polarisation_type='linear')
    expected = np.asarray([[1+2, 3+4*1j, 3-4*1j, 1-2],
                            [5+6,7+8*1j,7-8*1j,5-6]])
    assert np.all(B==expected)

    # Circular (I,Q,U,V)
    B = brightness(stokes, polarisation_type='circular')
    expected = np.asarray([[1+4, 2+3*1j, 2-3*1j, 1-4],
                            [5+8,6+7*1j,6-7*1j,5-8]])
    assert np.all(B==expected)

from africanus.stokes.dask import have_requirements

@pytest.mark.skipif(not have_requirements, reason="requirements not installed")
def test_dask_brightness():
    import dask.array as da
    import numpy as np
    from africanus.stokes.dask import brightness

    stokes = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float64)

    B = brightness(da.from_array(stokes, (1,4)), polarisation_type='linear')
    expected = np.asarray([[1+2, 3+4*1j, 3-4*1j, 1-2],
                            [5+6,7+8*1j,7-8*1j,5-6]])
    assert np.all(B.compute()==expected)

    B = brightness(da.from_array(stokes, (1,4)), polarisation_type='circular')
    expected = np.asarray([[1+4, 2+3*1j, 2-3*1j, 1-4],
                            [5+8,6+7*1j,6-7*1j,5-8]])
    assert np.all(B.compute()==expected)
