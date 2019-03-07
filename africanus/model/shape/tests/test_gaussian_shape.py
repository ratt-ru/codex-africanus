# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np

from africanus.model.shape import gaussian


def test_gauss_shape():
    row = 10
    chan = 16

    shape_params = np.array([[.4, .3, .2],
                             [.4, .3, .2]])
    uvw = np.random.random((row, 3))
    freq = np.linspace(.856e9, 2*.856e9, chan)

    gauss_shape = gaussian(uvw, freq, shape_params)

    assert gauss_shape.shape == (shape_params.shape[0], row, chan)
