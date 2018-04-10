#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

import numpy as np

import pytest

def test_phase_delay():
    from africanus.rime import phase_delay

    uvw = np.random.random(size=(100,3))
    lm = np.random.random(size=(10,2))
    frequency = np.linspace(.856e9, .856e9*2, 64, endpoint=True)

    from africanus.constants import minus_two_pi_over_c

    # Test complex phase at a particular index in the output
    uvw_i, lm_i, freq_i = 2, 3, 5

    u, v, w = [1,2,3]
    l, m = [0.1, 0.2]
    freq = 0.856e9

    # Set up values in the input
    uvw[uvw_i] = [u, v, w]
    lm[lm_i] = [l, m]
    frequency[freq_i] = freq

    # Compute complex phase
    complex_phase = phase_delay(uvw, lm, frequency)

    # Test singular value vs a point in the output
    n = np.sqrt(1.0 - l**2 - m**2) - 1.0
    phase = minus_two_pi_over_c*(u*l + v*m + w*n)*freq
    assert np.all(np.exp(1j*phase) == complex_phase[lm_i, uvw_i, freq_i])


from africanus.rime.dask import have_requirements

@pytest.mark.skipif(not have_requirements, reason="requirements not installed")
def test_dask_phase_delay():
    import dask.array as da
    from africanus.rime import phase_delay as np_phase_delay
    from africanus.rime.dask import phase_delay as dask_phase_delay

    uvw = np.random.random(size=(100,3))
    lm = np.random.random(size=(10,2))*0.01 # So that 1 > 1 - l**2 - m**2 >= 0
    frequency = np.linspace(.856e9, .856e9*2, 64, endpoint=True)

    dask_uvw = da.from_array(uvw, chunks=(25,3))
    dask_lm = da.from_array(lm, chunks=(5, 2))
    dask_frequency = da.from_array(frequency, chunks=16)

    dask_phase = dask_phase_delay(dask_uvw, dask_lm, dask_frequency).compute()
    np_phase = np_phase_delay(uvw, lm, frequency)

    # Should agree completely
    assert np.all(np_phase == dask_phase)




