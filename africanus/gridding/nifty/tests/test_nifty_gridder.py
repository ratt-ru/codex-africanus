# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import pytest

from africanus.gridding.nifty.dask import grid, grid_config, dirty


def rf(*a, **kw):
    return np.random.random(*a, **kw)


def rc(*a, **kw):
    return rf(*a, **kw) + 1j*rf(*a, **kw)


def test_dask_nifty_gridder():
    da = pytest.importorskip('dask.array')
    ng = pytest.importorskip('nifty_gridder')

    row = (16, 16, 16, 16)
    chan = (32,)
    corr = (4,)

    nrow = sum(row)
    nchan = sum(chan)
    ncorr = sum(corr)

    # Random UV data
    uvw = rf(size=(nrow, 3)).astype(np.float64)*128
    vis = rf(size=(nrow, nchan, ncorr)).astype(np.complex128)
    freq = np.linspace(.856e9, 2*.856e9, nchan)
    flag = np.zeros(vis.shape, dtype=np.uint8)
    weight = np.ones(vis.shape, dtype=np.float64)

    da_vis = da.from_array(vis, chunks=(row, chan, corr))
    da_uvw = da.from_array(uvw, chunks=(row, 3))
    da_freq = da.from_array(freq, chunks=chan)
    da_flag = da.from_array(flag, chunks=(row, chan, corr))
    da_weight = da.from_array(weight, chunks=(row, chan, corr))

    gc = grid_config(1024, 1024, 2.0, 2.0, 2e-13)
    g = grid(da_vis, da_uvw, da_flag, da_weight, da_freq, gc)
    d = dirty(g, gc)

    assert g.shape == (gc.Nu(), gc.Nv(), ncorr)
    assert d.shape == (gc.Nxdirty(), gc.Nydirty(), ncorr)

    d.compute()

