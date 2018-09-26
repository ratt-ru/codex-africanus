#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest


def rf(*a, **kw):
    return np.random.random(*a, **kw)


def rc(*a, **kw):
    return rf(*a, **kw) + 1j*rf(*a, **kw)


def test_transform_sources():
    from africanus.rime import transform_sources

    src = 10
    time = 5
    ants = 4
    chans = 8

    lm = np.random.random(size=(src, 2))
    parangles = np.random.random(size=(time, ants))
    point_errors = np.random.random(size=(time, ants, 2))
    antenna_scaling = np.random.random(size=(ants, chans))
    frequency = np.linspace(.856e9, .856e8*2, chans)

    coords = transform_sources(lm, parangles, point_errors,
                               antenna_scaling, frequency)

    assert coords.shape == (3, src, time, ants, chans)


def test_dask_transform_sources():
    da = pytest.importorskip("dask.array")

    from africanus.rime.dask import transform_sources as dask_xform_src
    from africanus.rime import transform_sources as np_xform_src

    src = 10
    src_chunks = (1, 2, 3, 4)
    time = 5
    time_chunks = (2, 3)
    ants = 4
    ant_chunks = (2, 2)
    chans = 8
    chan_chunks = (5, 3)

    lm = np.random.random(size=(src, 2))
    parangles = np.random.random(size=(time, ants))
    point_errors = np.random.random(size=(time, ants, 2))
    antenna_scaling = np.random.random(size=(ants, chans))
    frequency = np.linspace(.856e9, .856e8*2, chans)

    dask_lm = da.from_array(lm, chunks=(src_chunks, 2))
    dask_pa = da.from_array(parangles, chunks=(time_chunks, ant_chunks))
    dask_pe = da.from_array(point_errors, chunks=(time_chunks, ant_chunks, 2))
    dask_as = da.from_array(antenna_scaling, chunks=(ant_chunks, chan_chunks))
    dask_fq = da.from_array(frequency, chunks=(chan_chunks,))

    np_coords = np_xform_src(lm, parangles, point_errors,
                             antenna_scaling, frequency)
    dask_coords = dask_xform_src(dask_lm, dask_pa, dask_pe,
                                 dask_as, dask_fq)

    # Should agree exactly
    assert np.all(dask_coords.compute() == np_coords)


def test_beam_cube():
    beam_lw = 10
    beam_mh = 10
    beam_nud = 10

    src = 10
    time = 5
    ants = 4
    chans = 8

    # Source Transform variables
    lm = np.random.random(size=(src, 2))
    parangles = np.random.random(size=(time, ants))
    point_errors = np.random.random(size=(time, ants, 2))
    antenna_scaling = np.random.random(size=(ants, chans))
    freqs = np.linspace(.856e9, .856e9*2, chans)

    from africanus.rime import beam_cube_dde
    from africanus.rime import transform_sources

    coords = transform_sources(lm, parangles, point_errors,
                               antenna_scaling, freqs)

    # Beam sampling variables
    beam = rc((beam_lw, beam_mh, beam_nud, 2, 2))
    l_grid = np.linspace(-1, 1, beam_lw)
    m_grid = np.linspace(-1, 1, beam_mh)
    freq_grid = np.linspace(.856e9, .856e9*2, beam_nud)
    freq_grid[1:-1] += rf((beam_nud-2,))*1e5

    ddes = beam_cube_dde(beam, coords, l_grid, m_grid, freq_grid)
    assert ddes.shape == (src, time, ants, chans, 2, 2)


def test_dask_beam_cube():
    da = pytest.importorskip('dask.array')

    beam_lw = 10
    beam_mh = 10
    beam_nud = 10

    src = 10
    src_chunks = (1, 2, 3, 4)
    time = 5
    time_chunks = (2, 3)
    ants = 4
    ant_chunks = (2, 2)
    chans = 8
    chan_chunks = (5, 3)

    # Source Transform variables
    lm = np.random.random(size=(src, 2))
    parangles = np.random.random(size=(time, ants))
    point_errors = np.random.random(size=(time, ants, 2))
    antenna_scaling = np.random.random(size=(ants, chans))
    freqs = np.linspace(.856e9, .856e9*2, chans)

    # Beam sampling variables
    beam = rc((beam_lw, beam_mh, beam_nud, 2, 2))
    l_grid = np.linspace(-1, 1, beam_lw)
    m_grid = np.linspace(-1, 1, beam_mh)
    freq_grid = np.linspace(.856e9, .856e9*2, beam_nud)
    freq_grid[1:-1] += rf((beam_nud-2,))*1e5

    from africanus.rime import transform_sources as np_transform_sources
    from africanus.rime import beam_cube_dde as np_cube_dde

    # compute numpy coordinates and ddes
    np_coords = np_transform_sources(lm, parangles, point_errors,
                                     antenna_scaling, freqs)
    np_ddes = np_cube_dde(beam, np_coords, l_grid, m_grid, freq_grid)

    from africanus.rime.dask import transform_sources
    from africanus.rime.dask import beam_cube_dde

    # Dask source transform variables
    dask_lm = da.from_array(lm, chunks=(src_chunks, 2))
    dask_parangles = da.from_array(parangles, chunks=(time_chunks, ant_chunks))
    dask_point_errors = da.from_array(point_errors,
                                      chunks=(time_chunks, ant_chunks, 2))
    dask_antenna_scaling = da.from_array(antenna_scaling,
                                         chunks=(ant_chunks, chan_chunks))
    dask_freqs = da.from_array(freqs, chunks=chan_chunks)

    # Dask beam sampling variables
    dask_beam = da.from_array(beam, chunks=beam.shape)
    dask_l_grid = da.from_array(l_grid, chunks=l_grid.shape)
    dask_m_grid = da.from_array(m_grid, chunks=m_grid.shape)
    dask_freq_grid = da.from_array(freq_grid, chunks=freq_grid.shape)

    # Compute dask coordinates and ddes
    dask_coords = transform_sources(dask_lm, dask_parangles, dask_point_errors,
                                    dask_antenna_scaling, dask_freqs)

    ddes = beam_cube_dde(dask_beam, dask_coords,
                         dask_l_grid, dask_m_grid, dask_freq_grid)

    # Should agree exactly
    assert np.all(ddes.compute() == np_ddes)
