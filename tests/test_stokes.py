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

    B = brightness(stokes, polarisation_type='linear', corr_shape='matrix')
    assert np.all(B==expected.reshape(2,2,1))

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

    # Test correlation shape
    B = brightness(stokes, polarisation_type='linear', corr_shape='matrix')
    expected = np.asarray([[1+2, 3+4*1j, 3-4*1j, 1-2],
                            [5+6,7+8*1j,7-8*1j,5-6]]).reshape(2,2,2)
    assert np.all(B==expected)


def test_brightness_shape():
    import numpy as np
    from africanus.stokes import brightness

    for pol_type in ('linear', 'circular'):
        # 4 polarisation case
        assert brightness(np.random.random((10,5,3,4)),
            polarisation_type=pol_type,
            corr_shape='matrix').shape == (10,5,3,2,2)

        assert brightness(np.random.random((10,5,3,4)),
            polarisation_type=pol_type,
            corr_shape='flat').shape == (10,5,3,4)

        # 2 polarisation case
        assert brightness(np.random.random((10,5,3,2)),
            polarisation_type=pol_type,
            corr_shape='matrix').shape == (10,5,3,2,1)

        assert brightness(np.random.random((10,5,3,2)),
            polarisation_type=pol_type,
            corr_shape='flat').shape == (10,5,3,2)

        # 1 polarisation case
        assert brightness(np.random.random((10,5,3,1)),
            polarisation_type=pol_type,
            corr_shape='matrix').shape == (10,5,3,1)

        assert brightness(np.random.random((10,5,3,1)),
            polarisation_type=pol_type,
            corr_shape='flat').shape == (10,5,3,1)


from africanus.stokes.dask import have_requirements

@pytest.mark.skipif(not have_requirements, reason="requirements not installed")
def test_dask_brightness():
    import dask.array as da
    import numpy as np
    from africanus.stokes.dask import brightness

    stokes = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float64)
    da_stokes = da.from_array(stokes, (1,4))

    B = brightness(da_stokes, polarisation_type='linear')
    expected = np.asarray([[1+2, 3+4*1j, 3-4*1j, 1-2],
                            [5+6,7+8*1j,7-8*1j,5-6]])
    assert np.all(B.compute()==expected)

    # Test correlation shape
    B = brightness(da_stokes, polarisation_type='linear', corr_shape='matrix')
    assert np.all(B.compute()==expected.reshape(2,2,2))


    B = brightness(da_stokes, polarisation_type='circular')
    expected = np.asarray([[1+4, 2+3*1j, 2-3*1j, 1-4],
                            [5+8,6+7*1j,6-7*1j,5-8]])

    # Test correlation shape
    B = brightness(da_stokes, polarisation_type='circular', corr_shape='matrix')
    assert np.all(B.compute()==expected.reshape(2,2,2))


@pytest.mark.skipif(not have_requirements, reason="requirements not installed")
def test_dask_brightness_shape():
    import dask.array as da
    from africanus.stokes.dask import brightness

    for pol_type in ('linear', 'circular'):
        # 4 polarisation case
        stokes = da.random.random((10,5,3,4),chunks=(5,(2,3),(2,1),4))

        assert brightness(stokes,
            polarisation_type=pol_type,
            corr_shape='matrix').compute().shape == (10,5,3,2,2)

        assert brightness(stokes,
            polarisation_type=pol_type,
            corr_shape='flat').shape == (10,5,3,4)

        # 2 polarisation case
        stokes = da.random.random((10,5,3,2),chunks=(5,(2,3),(2,1),2))

        assert brightness(stokes,
            polarisation_type=pol_type,
            corr_shape='matrix').shape == (10,5,3,2,1)

        assert brightness(stokes,
            polarisation_type=pol_type,
            corr_shape='flat').shape == (10,5,3,2)

        # 1 polarisation case
        stokes = da.random.random((10,5,3,1),chunks=(5,(2,3),(2,1),1))

        assert brightness(stokes,
            polarisation_type=pol_type,
            corr_shape='matrix').shape == (10,5,3,1)

        assert brightness(stokes,
            polarisation_type=pol_type,
            corr_shape='flat').shape == (10,5,3,1)
