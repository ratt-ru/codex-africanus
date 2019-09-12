# -*- coding: utf-8 -*-


import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from africanus.model.shape import gaussian as np_gaussian


def test_gauss_shape():
    row = 10
    chan = 16

    shape_params = np.array([[.4, .3, .2],
                             [.4, .3, .2]])
    uvw = np.random.random((row, 3))
    freq = np.linspace(.856e9, 2*.856e9, chan)

    gauss_shape = np_gaussian(uvw, freq, shape_params)

    assert gauss_shape.shape == (shape_params.shape[0], row, chan)


def test_dask_gauss_shape():
    da = pytest.importorskip('dask.array')
    from africanus.model.shape.dask import gaussian as da_gaussian

    row_chunks = (5, 5)
    chan_chunks = (4, 4)

    row = sum(row_chunks)
    chan = sum(chan_chunks)

    shape_params = da.asarray([[.4, .3, .2],
                               [.4, .3, .2]])
    uvw = da.random.random((row, 3), chunks=(row_chunks, 3))
    freq = da.linspace(.856e9, 2*.856e9, chan, chunks=chan_chunks)
    da_gauss_shape = da_gaussian(uvw, freq, shape_params).compute()
    np_gauss_shape = np_gaussian(uvw.compute(),
                                 freq.compute(),
                                 shape_params.compute())

    assert_array_almost_equal(da_gauss_shape, np_gauss_shape)
