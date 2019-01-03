#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

import numpy as np

import pytest


from africanus.stokes.stokes_conversion import (
                stokes_convert as np_stokes_convert,
                STOKES_TYPE_MAP as smap)


stokes_corr_cases = [
    ("complex", [['XX'], ['YY']],
     "real", ['I', 'Q']),
    ("complex", ['XX', 'YY'],
     "real", ['I', 'Q']),
    ("complex", ['XX', 'XY', 'YX', 'YY'],
     "real", ['I', 'Q', 'U', 'V']),
    ("complex", [['XX', 'XY'], ['YX', 'YY']],
     "real", [['I', 'Q'], ['U', 'V']]),
    ("real", ['I', 'Q', 'U', 'V'],
     "complex", ['XX', 'XY', 'YX', 'YY']),
    ("real", [['I', 'Q'], ['U', 'V']],
     "complex", [['XX', 'XY'], ['YX', 'YY']]),
    ("real", [['I', 'Q'], ['U', 'V']],
     "complex", [['XX', 'XY', 'YX', 'YY']]),
    ("real", [['I', 'Q'], ['U', 'V']],
     "complex", [['RR', 'RL', 'LR', 'LL']]),
    ("real", ['I', 'V'],
     "complex", ['RR', 'LL']),
    ("real", ['I', 'Q'],
     "complex", ['XX', 'YY']),
]

stokes_corr_int_cases = [
    ("complex", [smap['XX'], smap['YY']],
     "real", [smap['I'], smap['Q']])
]


def visibility_factory(vis_shape, input_shape, in_type,
                       backend="numpy", **kwargs):
    shape = vis_shape + input_shape

    if backend == "numpy":
        vis = np.arange(1.0, np.product(shape) + 1.0)
        vis = vis.reshape(shape)
    elif backend == "dask":
        da = pytest.importorskip('dask.array')
        vis = da.arange(1.0, np.product(shape) + 1.0, chunks=np.product(shape))
        vis = vis.reshape(shape)
        vis = vis.rechunk(kwargs['vis_chunks'] + input_shape)
    else:
        raise ValueError("Invalid backend %s" % backend)

    if in_type == "real":
        pass
    elif in_type == "complex":
        vis = vis + 1j*vis
    else:
        raise ValueError("Invalid in_type %s" % in_type)

    return vis


@pytest.mark.parametrize("in_type, input_schema, out_type, output_schema",
                         stokes_corr_cases + stokes_corr_int_cases)
@pytest.mark.parametrize("vis_shape", [
    (10, 5, 3),
    (6, 8),
    (15,),
])
def test_stokes_schemas(in_type, input_schema,
                        out_type, output_schema,
                        vis_shape):
    input_shape = np.asarray(input_schema).shape
    output_shape = np.asarray(output_schema).shape
    vis = visibility_factory(vis_shape, input_shape, in_type)
    xformed_vis = np_stokes_convert(vis, input_schema, output_schema)
    assert xformed_vis.shape == vis_shape + output_shape


def test_stokes_conversion():
    I, Q, U, V = [1.0, 2.0, 3.0, 4.0]

    # Check conversion to linear (string)
    vis = np_stokes_convert(np.asarray([[I, Q, U, V]]),
                            ['I', 'Q', 'U', 'V'],
                            ['XX', 'XY', 'YX', 'YY'])

    XX, XY, YX, YY = vis[0]
    assert np.all(vis == [[I + Q, U + V*1j, U - V*1j, I - Q]])

    # Check conversion to linear (integer)
    vis = np_stokes_convert(np.asarray([[I, Q, U, V]]),
                            [smap[x] for x in ('I', 'Q', 'U', 'V')],
                            [smap[x] for x in ('XX', 'XY', 'YX', 'YY')])

    assert np.all(vis == [[I + Q, U + V*1j, U - V*1j, I - Q]])

    # Check conversion to circular (string)
    vis = np_stokes_convert(np.asarray([[I, Q, U, V]]),
                            ['I', 'Q', 'U', 'V'],
                            ['RR', 'RL', 'LR', 'LL'])

    RR, RL, LR, LL = vis[0]
    assert np.all(vis == [[I + V, Q + U*1j, Q - U*1j, I - V]])

    # Check conversion to circular (integer)
    vis = np_stokes_convert(np.asarray([[I, Q, U, V]]),
                            [smap[x] for x in ('I', 'Q', 'U', 'V')],
                            [smap[x] for x in ('RR', 'RL', 'LR', 'LL')])

    assert np.all(vis == [[I + V, Q + U*1j, Q - U*1j, I - V]])

    # linear to stokes (string)
    stokes = np_stokes_convert(np.asarray([[XX, XY, YX, YY]]),
                               ['XX', 'XY', 'YX', 'YY'],
                               ['I', 'Q', 'U', 'V'])

    assert np.all(stokes == [[I, Q, U, V]])

    # linear to stokes  (integer)
    stokes = np_stokes_convert(np.asarray([[XX, XY, YX, YY]]),
                               [smap[x] for x in ('XX', 'XY', 'YX', 'YY')],
                               [smap[x] for x in ('I', 'Q', 'U', 'V')])

    assert np.all(stokes == [[I, Q, U, V]])

    # circular to stokes (string)
    stokes = np_stokes_convert(np.asarray([[RR, RL, LR, LL]]),
                               ['RR', 'RL', 'LR', 'LL'],
                               ['I', 'Q', 'U', 'V'])

    assert np.all(stokes == [[I, Q, U, V]])

    # circular to stokes (intger)
    stokes = np_stokes_convert(np.asarray([[RR, RL, LR, LL]]),
                               [smap[x] for x in ('RR', 'RL', 'LR', 'LL')],
                               [smap[x] for x in ('I', 'Q', 'U', 'V')])

    assert np.all(stokes == [[I, Q, U, V]])


@pytest.mark.parametrize("in_type, input_schema, out_type, output_schema",
                         stokes_corr_cases + stokes_corr_int_cases)
@pytest.mark.parametrize("vis_chunks", [
    ((10, 5, 3), (2, 3), (3,)),
    ((6, 8), (3, 3), (4, 4)),
    ((5, 5, 5),),
])
def test_dask_stokes_conversion(in_type, input_schema,
                                out_type, output_schema,
                                vis_chunks):
    from africanus.stokes.dask import stokes_convert as da_stokes_convert

    vis_shape = tuple(sum(dim_chunks) for dim_chunks in vis_chunks)
    input_shape = np.asarray(input_schema).shape
    vis = visibility_factory(vis_shape, input_shape, in_type,
                             backend="dask", vis_chunks=vis_chunks)

    da_vis = da_stokes_convert(vis, input_schema, output_schema)
    np_vis = np_stokes_convert(vis.compute(), input_schema, output_schema)
    assert np.all(da_vis == np_vis)
