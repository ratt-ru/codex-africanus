# -*- coding: utf-8 -*-


import numpy as np
from numpy.testing import assert_array_almost_equal
import pickle
import pytest

from africanus.gridding.nifty.dask import (grid, degrid, dirty, model,
                                           grid_config)


def rf(*a, **kw):
    return np.random.random(*a, **kw)


def rc(*a, **kw):
    return rf(*a, **kw) + 1j*rf(*a, **kw)


def test_dask_nifty_gridder():
    """ Only tests that we can call it and create a dirty image """
    dask = pytest.importorskip('dask')
    da = pytest.importorskip('dask.array')
    _ = pytest.importorskip('nifty_gridder')

    row = (16,)*8
    chan = (32,)
    corr = (4,)
    nx = 1026
    ny = 1022

    nrow = sum(row)
    nchan = sum(chan)
    ncorr = sum(corr)

    # Random UV data
    uvw = rf(size=(nrow, 3)).astype(np.float64)*128
    vis = rf(size=(nrow, nchan, ncorr)).astype(np.complex128)
    freq = np.linspace(.856e9, 2*.856e9, nchan)
    flag = np.zeros(vis.shape, dtype=np.uint8)
    flag = np.random.randint(0, 2, vis.shape, dtype=np.uint8).astype(np.bool)
    weight = rf(vis.shape).astype(np.float64)

    da_vis = da.from_array(vis, chunks=(row, chan, corr))
    da_uvw = da.from_array(uvw, chunks=(row, 3))
    da_freq = da.from_array(freq, chunks=chan)
    da_flag = da.from_array(flag, chunks=(row, chan, corr))
    da_weight = da.from_array(weight, chunks=(row, chan, corr))

    gc = grid_config(nx, ny, 2e-13, 2.0, 2.0)

    # Standard fan reduction
    g = grid(da_vis, da_uvw, da_flag, da_weight, da_freq, gc)
    d1 = dirty(g, gc)

    grid_shape = (gc.object.Nu(), gc.object.Nv(), ncorr)
    dirty_shape = (gc.object.Nxdirty(), gc.object.Nydirty(), ncorr)

    assert g.shape == grid_shape
    assert d1.shape == dirty_shape == (nx, ny, ncorr)

    # Stream reduction (single stream)
    g = grid(da_vis, da_uvw, da_flag, da_weight, da_freq, gc, streams=1)
    d2 = dirty(g, gc)

    assert g.shape == grid_shape
    assert d2.shape == dirty_shape == (nx, ny, ncorr)

    # Stream reductino (three streams)
    g = grid(da_vis, da_uvw, da_flag, da_weight, da_freq, gc, streams=3)
    d3 = dirty(g, gc)

    assert g.shape == grid_shape
    assert d3.shape == dirty_shape == (nx, ny, ncorr)

    # All three techniques produce similar results
    d1, d2, d3 = dask.compute(d1, d2, d3)
    assert_array_almost_equal(d1, d2)
    assert_array_almost_equal(d2, d3)


def test_dask_nifty_degridder():
    """ Only tests that we can call it and create some visibilities """
    da = pytest.importorskip('dask.array')
    _ = pytest.importorskip('nifty_gridder')

    row = (16, 16, 16, 16)
    chan = (32,)
    corr = (4,)

    nrow = sum(row)
    nchan = sum(chan)
    ncorr = sum(corr)
    nx = 1026
    ny = 1022

    gc = grid_config(nx, ny, 2e-13, 2.0, 2.0)

    # Random UV data
    uvw = rf(size=(nrow, 3)).astype(np.float64)*128
    freq = np.linspace(.856e9, 2*.856e9, nchan)
    flag = np.zeros((nrow, nchan, ncorr), dtype=np.bool)
    weight = np.ones((nrow, nchan, ncorr), dtype=np.float64)
    image = rc(size=(nx, ny, ncorr)).astype(np.complex128)

    da_uvw = da.from_array(uvw, chunks=(row, 3))
    da_freq = da.from_array(freq, chunks=chan)
    da_flag = da.from_array(flag, chunks=(row, chan, corr))
    da_weight = da.from_array(weight, chunks=(row, chan, corr))
    da_image = da.from_array(image, chunks=(nx, ny, 1))

    da_grid_vis = model(da_image, gc)
    da_vis = degrid(da_grid_vis, da_uvw, da_flag, da_weight, da_freq, gc)
    vis = da_vis.compute()
    assert vis.shape == da_vis.shape


def test_pickle_gridder_config():
    gc = grid_config(512, 1024, 5e-13, 1.3, 2.0)
    gc2 = pickle.loads(pickle.dumps(gc))
    assert gc is not gc2
    assert gc.object.Nxdirty() == gc2.object.Nxdirty() == 512
    assert gc.object.Nydirty() == gc2.object.Nydirty() == 1024
    assert gc.object.Epsilon() == gc2.object.Epsilon() == 5e-13
    assert gc.object.Pixsize_x() == gc2.object.Pixsize_x() == 1.3
    assert gc.object.Pixsize_y() == gc2.object.Pixsize_y() == 2.0
