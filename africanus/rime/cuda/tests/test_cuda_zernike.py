# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from africanus.rime.cuda.zernike import zernike_dde


def rf(*args, **kwargs):
    return np.random.random(*args, **kwargs)


def rc(*args, **kwargs):
    return rf(*args, **kwargs) + 1j*rf(*args, **kwargs)


def test_cuda_zernike():
    cp = pytest.importorskip('cupy')

    src = 5
    time = 10
    ant = 7
    chan = 16
    corrs = (2, 2)
    npoly = 16

    coords = rf((3, src, time, ant, chan))
    coeffs = rf((ant, chan, npoly) + corrs)
    noll_index = rf((ant, chan, npoly) + corrs).astype(np.int32)

    ddes = zernike_dde(cp.asarray(coords),
                       cp.asarray(coeffs),
                       cp.asarray(noll_index))
