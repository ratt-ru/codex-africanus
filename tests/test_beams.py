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
    frequency = np.random.random(size=(chans,))

    coords = transform_sources(lm, parangles, point_errors,
                antenna_scaling, frequency)

    assert coords.shape == (3, src, time, ants, chans)


def test_dask_transform_sources():
    import dask.array as da

    from africanus.ddes.dask import transform_sources as dask_xform_src
    from africanus.ddes import transform_sources as np_xform_src

    src = 10
    src_chunks = (1,2,3,4)
    time = 5
    time_chunks = (2,3)
    ants = 4
    ant_chunks = (2,2)
    chans = 8
    chan_chunks = (5,3)

    lm = np.random.random(size=(src,2))
    parangles = np.random.random(size=(time, ants))
    point_errors = np.random.random(size=(time, ants, 2))
    antenna_scaling = np.random.random(size=(ants, chans))
    frequency = np.linspace(.856e9, .856e8*2, chans)

    dask_lm = da.from_array(lm, chunks=(src_chunks, 2))
    dask_pa = da.from_array(parangles, chunks=(time_chunks,ant_chunks))
    dask_pe = da.from_array(point_errors, chunks=(time_chunks,ant_chunks,2))
    dask_as = da.from_array(antenna_scaling, chunks=(ant_chunks, chan_chunks))
    dask_fq = da.from_array(frequency, chunks=(chan_chunks,))

    np_coords = np_xform_src(lm, parangles, point_errors,
                                antenna_scaling, frequency)
    dask_coords = dask_xform_src(dask_lm, dask_pa, dask_pe,
                                dask_as, dask_fq)

    # Should agree exactly
    assert np.all(dask_coords.compute() == np_coords)


