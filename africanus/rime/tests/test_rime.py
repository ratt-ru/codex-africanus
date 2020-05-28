#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

import pytest


def rf(*a, **kw):
    return np.random.random(*a, **kw)


def rc(*a, **kw):
    return rf(*a, **kw) + 1j*rf(*a, **kw)


@pytest.mark.parametrize("convention, sign",  [
    ('fourier', 1),
    ('casa', -1)
])
def test_phase_delay(convention, sign):
    from africanus.rime import phase_delay

    uvw = np.random.random(size=(100, 3))
    lm = np.random.random(size=(10, 2))
    frequency = np.linspace(.856e9, .856e9*2, 64, endpoint=True)

    from africanus.constants import minus_two_pi_over_c

    # Test complex phase at a particular index in the output
    uvw_i, lm_i, freq_i = 2, 3, 5

    u, v, w = [1, 2, 3]
    l, m = [0.1, 0.2]
    freq = 0.856e9

    # Set up values in the input
    uvw[uvw_i] = [u, v, w]
    lm[lm_i] = [l, m]
    frequency[freq_i] = freq

    # Compute complex phase
    complex_phase = phase_delay(lm, uvw, frequency, convention=convention)

    # Test singular value vs a point in the output
    n = np.sqrt(1.0 - l**2 - m**2) - 1.0
    phase = sign*minus_two_pi_over_c*(u*l + v*m + w*n)*freq
    assert np.all(np.exp(1j*phase) == complex_phase[lm_i, uvw_i, freq_i])


@pytest.mark.parametrize("second_rotation_angle", [False, True])
def test_feed_rotation(second_rotation_angle):
    from africanus.rime import feed_rotation

    shape = (10, 4)

    parangles = np.random.random(shape)
    pa_sin = np.sin(parangles)
    pa_cos = np.cos(parangles)

    if second_rotation_angle:
        parangles2 = np.random.random(shape)
        pa2_sin = np.sin(parangles2)
        pa2_cos = np.cos(parangles2)
    else:
        parangles2 = parangles
        pa2_sin = pa_sin
        pa2_cos = pa_cos

    args = (parangles, parangles2) if second_rotation_angle else (parangles,)

    fr = feed_rotation(*args, feed_type='linear')
    np_expr = np.stack([pa_cos, pa_sin, -pa2_sin, pa2_cos], axis=2)
    assert_array_almost_equal(fr, np_expr.reshape(shape + (2, 2)))

    fr = feed_rotation(*args, feed_type='circular')
    zeros = np.zeros_like(pa_sin)
    np_expr = np.stack([pa_cos - 1j*pa_sin, zeros,
                        zeros, pa2_cos + 1j*pa2_sin], axis=2)
    assert_array_almost_equal(fr, np_expr.reshape(shape + (2, 2)))


@pytest.mark.parametrize("convention, sign",  [
    ('fourier', 1),
    ('casa', -1)
])
def test_dask_phase_delay(convention, sign):
    da = pytest.importorskip('dask.array')
    from africanus.rime import phase_delay as np_phase_delay
    from africanus.rime.dask import phase_delay as dask_phase_delay

    # So that 1 > 1 - l**2 - m**2 >= 0
    lm = np.random.random(size=(10, 2))*0.01
    uvw = np.random.random(size=(100, 3))
    frequency = np.linspace(.856e9, .856e9*2, 64, endpoint=True)

    dask_lm = da.from_array(lm, chunks=(5, 2))
    dask_uvw = da.from_array(uvw, chunks=(25, 3))
    dask_frequency = da.from_array(frequency, chunks=16)

    dask_phase = dask_phase_delay(dask_lm, dask_uvw, dask_frequency,
                                  convention=convention)
    np_phase = np_phase_delay(lm, uvw, frequency, convention=convention)

    # Should agree completely
    assert np.all(np_phase == dask_phase.compute())


@pytest.mark.parametrize("second_rotation_angle", [False, True])
def test_dask_feed_rotation(second_rotation_angle):
    da = pytest.importorskip('dask.array')
    from africanus.rime import feed_rotation as np_feed_rotation
    from africanus.rime.dask import feed_rotation

    chunks = ((5, 5), (2, 3))
    shape = tuple(map(sum, chunks))

    rot_angles = np.random.random(shape)
    dask_rot_angles = da.from_array(rot_angles, chunks=chunks)

    if second_rotation_angle:
        rot_angles2 = np.random.random(shape)
        dask_rot_angles2 = da.from_array(rot_angles2, chunks=chunks)
        np_args = (rot_angles, rot_angles2)
        dsk_args = (dask_rot_angles, dask_rot_angles2)
    else:
        np_args = (rot_angles,)
        dsk_args = (dask_rot_angles,)

    np_fr = np_feed_rotation(*np_args, feed_type='linear')
    assert_array_equal(np_fr, feed_rotation(*dsk_args, feed_type='linear'))

    np_fr = np_feed_rotation(*np_args, feed_type='circular')
    assert_array_equal(np_fr, feed_rotation(*dsk_args, feed_type='circular'))
