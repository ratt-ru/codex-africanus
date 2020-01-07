#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

import numpy as np

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


def test_feed_rotation():
    import numpy as np
    from africanus.rime import feed_rotation

    parangles = np.random.random((10, 5))
    pa_sin = np.sin(parangles)
    pa_cos = np.cos(parangles)

    fr = feed_rotation(parangles, feed_type='linear')
    np_expr = np.stack([pa_cos, pa_sin, -pa_sin, pa_cos], axis=2)
    assert np.allclose(fr, np_expr.reshape(10, 5, 2, 2))

    fr = feed_rotation(parangles, feed_type='circular')
    zeros = np.zeros_like(pa_sin)
    np_expr = np.stack([pa_cos - 1j*pa_sin, zeros,
                        zeros, pa_cos + 1j*pa_sin], axis=2)
    assert np.allclose(fr, np_expr.reshape(10, 5, 2, 2))


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


def test_dask_feed_rotation():
    da = pytest.importorskip('dask.array')
    import numpy as np
    from africanus.rime import feed_rotation as np_feed_rotation
    from africanus.rime.dask import feed_rotation

    parangles = np.random.random((10, 5))
    dask_parangles = da.from_array(parangles, chunks=(5, (2, 3)))

    np_fr = np_feed_rotation(parangles, feed_type='linear')
    assert np.all(np_fr == feed_rotation(dask_parangles, feed_type='linear'))

    np_fr = np_feed_rotation(parangles, feed_type='circular')
    assert np.all(np_fr == feed_rotation(dask_parangles, feed_type='circular'))
