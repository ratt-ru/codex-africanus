#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

import numpy as np

import pytest

def test_dft():
    from africanus.dft import dft

    uvw = np.random.random(size=(100,3))
    lm = np.random.random(size=(10,2))
    frequency = np.linspace(.856e9, .856e9*2, 64, endpoint=True)

    complex_phase = dft(uvw, lm, frequency)

from africanus.dft.dask import have_requirements

@pytest.mark.skipif(not have_requirements, reason="requirements not installed")
def test_dask_dft():
    import dask.array as da
    from africanus.dft import dft as np_dft
    from africanus.dft.dask import dft as dask_dft

    uvw = np.random.random(size=(100,3))
    lm = np.random.random(size=(10,2))*0.01 # So that 1 > 1 - l**2 - m**2 >= 0
    frequency = np.linspace(.856e9, .856e9*2, 64, endpoint=True)

    dask_uvw = da.from_array(uvw, chunks=(25,3))
    dask_lm = da.from_array(lm, chunks=(5, 2))
    dask_frequency = da.from_array(frequency, chunks=16)

    dask_phase = dask_dft(dask_uvw, dask_lm, dask_frequency).compute()
    np_phase = np_dft(uvw, lm, frequency)

    # Should agree completely
    assert np.all(np_phase == dask_phase)




