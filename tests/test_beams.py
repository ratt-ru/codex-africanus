#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

def test_transform_sources():
    from africanus.ddes import transform_sources

    src = 10
    time = 5
    ants = 4
    chans = 8

    lm = np.random.random(size=(src,2))
    parangles = np.random.random(size=(time, ants))
    point_errors = np.random.random(size=(time, ants, 2))
    antenna_scaling = np.random.random(size=(ants, chans))

    coords = transform_sources(lm, parangles, point_errors, antenna_scaling)

    assert coords.shape == (src, time, ants, chans, 2)

def test_dask_transform_sources():
    import dask.array as da

    from africanus.ddes.dask import transform_sources as dask_xform_src
    from africanus.ddes import transform_sources as np_xform_src

    src = 10
    time = 5
    ants = 4
    chans = 8

    lm = np.random.random(size=(src,2))
    parangles = np.random.random(size=(time, ants))
    point_errors = np.random.random(size=(time, ants, 2))
    antenna_scaling = np.random.random(size=(ants, chans))

    dask_lm = da.from_array(lm, chunks=(5,2))
    dask_pa = da.from_array(parangles, chunks=((2,3),2))
    dask_pe = da.from_array(point_errors, chunks=((2,3),2,2))
    dask_as = da.from_array(antenna_scaling, chunks=(2, 4))

    dask_coords = dask_xform_src(dask_lm, dask_pa, dask_pe, dask_as)
    np_coords = np_xform_src(lm, parangles, point_errors, antenna_scaling)


    assert np.all(dask_coords.compute() == np_coords)


