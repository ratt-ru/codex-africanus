#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

import numpy as np

import pytest


from africanus.stokes.stokes_conversion import (
                stokes_convert as np_stokes_convert)


_stokes_corr_cases = [
    ([['XX'], ['YY']], ['I', 'Q']),
    (['XX', 'YY'], ['I', 'Q']),
    (['XX', 'XY', 'YX', 'YY'], ['I', 'Q', 'U', 'V']),
    ([['XX', 'XY'], ['YX', 'YY']], [['I', 'Q'], ['U', 'V']]),
    (['I', 'Q', 'U', 'V'], ['XX', 'XY', 'YX', 'YY']),
    ([['I', 'Q'], ['U', 'V']], [['XX', 'XY'], ['YX', 'YY']]),
    ([['I', 'Q'], ['U', 'V']], [['XX', 'XY', 'YX', 'YY']]),
    ([['I', 'Q'], ['U', 'V']], [['RR', 'RL', 'LR', 'LL']]),
    (['I', 'V'], ['RR', 'LL']),
    (['I', 'Q'], ['XX', 'YY']),
]


@pytest.mark.parametrize("input_schema, output_schema", _stokes_corr_cases)
@pytest.mark.parametrize("vis_shape", [
    (10, 5, 3),
    (6, 8),
    (15,),
])
def test_stokes_schemas(input_schema, output_schema, vis_shape):
    input_shape = np.asarray(input_schema).shape
    output_shape = np.asarray(output_schema).shape

    shape = vis_shape + input_shape

    vis = np.arange(1.0, np.product(shape) + 1.0)
    vis = vis.reshape(shape)
    vis = vis + vis*1j

    xformed_vis = np_stokes_convert(vis, input_schema, output_schema)
    assert xformed_vis.shape == vis_shape + output_shape


def test_stokes_conversion():
    I, Q, U, V = [1.0, 2.0, 3.0, 4.0]

    # Check conversion to linear
    vis = np_stokes_convert(np.asarray([[I, Q, U, V]]),
                            ['I', 'Q', 'U', 'V'],
                            ['XX', 'XY', 'YX', 'YY'])

    XX, XY, YX, YY = vis[0]
    assert np.all(vis == [[I + Q, U + V*1j, U - V*1j, I - Q]])

    # Check conversion to circular
    vis = np_stokes_convert(np.asarray([[I, Q, U, V]]),
                            ['I', 'Q', 'U', 'V'],
                            ['RR', 'RL', 'LR', 'LL'])

    RR, RL, LR, LL = vis[0]
    assert np.all(vis == [[I + V, Q + U*1j, Q - U*1j, I - V]])

    # linear to stokes
    stokes = np_stokes_convert(np.asarray([[XX, XY, YX, YY]]),
                               ['XX', 'XY', 'YX', 'YY'],
                               ['I', 'Q', 'U', 'V'])

    assert np.all(stokes == [[I, Q, U, V]])

    # circular to stokes
    stokes = np_stokes_convert(np.asarray([[RR, RL, LR, LL]]),
                               ['RR', 'RL', 'LR', 'LL'],
                               ['I', 'Q', 'U', 'V'])

    assert np.all(stokes == [[I, Q, U, V]])


@pytest.mark.parametrize("input_schema, output_schema", _stokes_corr_cases)
@pytest.mark.parametrize("vis_shape, vis_chunks", [
    ((10, 5, 3), (5, (2, 3), 3)),
    ((6, 8), (3, 4)),
    ((15,), (5, 5, 5)),
])
def test_dask_stokes_conversion(input_schema, output_schema,
                                vis_shape, vis_chunks):
    da = pytest.importorskip('dask.array')

    from africanus.stokes.dask import stokes_convert as da_stokes_convert

    input_shape = np.asarray(input_schema).shape
    shape = vis_shape + input_shape

    vis = da.arange(1.0, np.product(shape) + 1.0, chunks=np.product(shape))
    vis = vis.reshape(shape)
    vis = vis.rechunk(vis_chunks + input_shape)
    vis = vis + vis*1j

    da_vis = da_stokes_convert(vis, input_schema, output_schema)
    np_vis = np_stokes_convert(vis.compute(), input_schema, output_schema)
    assert np.all(da_vis == np_vis)
