#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

import numpy as np
import pytest

from africanus.model.coherency.conversion import convert as np_convert
from africanus.util.casa_types import STOKES_TYPE_MAP as smap

stokes_corr_cases = [
    ([["XX"], ["YY"]], ["I", "Q"]),
    (["XX", "YY"], ["I", "Q"]),
    (["XX", "XY", "YX", "YY"], ["I", "Q", "U", "V"]),
    ([["XX", "XY"], ["YX", "YY"]], [["I", "Q"], ["U", "V"]]),
    (["I", "Q", "U", "V"], ["XX", "XY", "YX", "YY"]),
    ([["I", "Q"], ["U", "V"]], [["XX", "XY"], ["YX", "YY"]]),
    ([["I", "Q"], ["U", "V"]], [["XX", "XY", "YX", "YY"]]),
    ([["I", "Q"], ["U", "V"]], [["RR", "RL", "LR", "LL"]]),
    (["I", "V"], ["RR", "LL"]),
    (["I", "Q"], ["XX", "YY"]),
]

stokes_corr_int_cases = [([smap["XX"], smap["YY"]], [smap["I"], smap["Q"]])]


vis_chunks = [
    ((10, 5, 3), (2, 3), (3,)),
    ((6, 8), (3, 3), (4, 4)),
    ((5, 5, 5),),
]


vis_shape = [tuple(sum(dim_chunks) for dim_chunks in case) for case in vis_chunks]


def visibility_factory(vis_shape, input_shape, backend="numpy", **kwargs):
    shape = vis_shape + input_shape

    if backend == "numpy":
        vis = np.arange(1.0, np.prod(shape) + 1.0)
        vis = vis.reshape(shape)
    elif backend == "dask":
        da = pytest.importorskip("dask.array")
        vis = da.arange(1.0, np.prod(shape) + 1.0, chunks=np.prod(shape))
        vis = vis.reshape(shape)
        vis = vis.rechunk(kwargs["vis_chunks"] + input_shape)
    else:
        raise ValueError("Invalid backend %s" % backend)

    return vis


@pytest.mark.parametrize(
    "input_schema, output_schema",
    stokes_corr_cases + stokes_corr_int_cases,
)
@pytest.mark.parametrize("vis_shape", vis_shape)
def test_conversion_schemas(input_schema, output_schema, vis_shape):
    input_shape = np.asarray(input_schema).shape
    output_shape = np.asarray(output_schema).shape
    vis = visibility_factory(vis_shape, input_shape)
    xformed_vis = np_convert(vis, input_schema, output_schema)
    assert xformed_vis.shape == vis_shape + output_shape


def test_conversion():
    I, Q, U, V = [1.0 + 1j, 2.0 + 2j, 3.0 + 3j, 4.0 + 4j]

    # Check conversion to linear (string)
    vis = np_convert(
        np.asarray([[I, Q, U, V]]), ["I", "Q", "U", "V"], ["XX", "XY", "YX", "YY"]
    )

    XX, XY, YX, YY = vis[0]
    assert np.all(vis == [[I + Q, U + V * 1j, U - V * 1j, I - Q]])

    # Check conversion to linear (integer)
    vis = np_convert(
        np.asarray([[I, Q, U, V]]),
        [smap[x] for x in ("I", "Q", "U", "V")],
        [smap[x] for x in ("XX", "XY", "YX", "YY")],
    )

    assert np.all(vis == [[I + Q, U + V * 1j, U - V * 1j, I - Q]])

    # Check conversion to circular (string)
    vis = np_convert(
        np.asarray([[I, Q, U, V]]), ["I", "Q", "U", "V"], ["RR", "RL", "LR", "LL"]
    )

    RR, RL, LR, LL = vis[0]
    assert np.all(vis == [[I + V, Q + U * 1j, Q - U * 1j, I - V]])

    # Check conversion to circular (integer)
    vis = np_convert(
        np.asarray([[I, Q, U, V]]),
        [smap[x] for x in ("I", "Q", "U", "V")],
        [smap[x] for x in ("RR", "RL", "LR", "LL")],
    )

    assert np.all(vis == [[I + V, Q + U * 1j, Q - U * 1j, I - V]])

    # linear to stokes (string)
    stokes = np_convert(
        np.asarray([[XX, XY, YX, YY]]), ["XX", "XY", "YX", "YY"], ["I", "Q", "U", "V"]
    )

    assert np.all(stokes == [[I, Q, U, V]])

    # linear to stokes  (integer)
    stokes = np_convert(
        np.asarray([[XX, XY, YX, YY]]),
        [smap[x] for x in ("XX", "XY", "YX", "YY")],
        [smap[x] for x in ("I", "Q", "U", "V")],
    )

    assert np.all(stokes == [[I, Q, U, V]])

    # circular to stokes (string)
    stokes = np_convert(
        np.asarray([[RR, RL, LR, LL]]), ["RR", "RL", "LR", "LL"], ["I", "Q", "U", "V"]
    )

    assert np.all(stokes == [[I, Q, U, V]])

    # circular to stokes (integer)
    stokes = np_convert(
        np.asarray([[RR, RL, LR, LL]]),
        [smap[x] for x in ("RR", "RL", "LR", "LL")],
        [smap[x] for x in ("I", "Q", "U", "V")],
    )

    assert np.all(stokes == [[I, Q, U, V]])

    # linear I to all
    vis = np_convert(
        np.asarray([I]), ["I"], ["XX", "XY", "YX", "YY"], implicit_stokes=True
    )

    assert vis[0] == I and vis[-1] == I

    # circular I to all
    vis = np_convert(
        np.asarray([I]), ["I"], ["RR", "RL", "LR", "LL"], implicit_stokes=True
    )

    assert vis[0] == I and vis[-1] == I


@pytest.mark.xfail
def test_convert_should_fail():
    # this should fail since I can't be computed from XX
    vis = np_convert(np.array([1.0 + 1j]), ["XX"], ["I"], implicit_stokes=True)


@pytest.mark.parametrize(
    "input_schema, output_schema",
    stokes_corr_cases + stokes_corr_int_cases,
)
@pytest.mark.parametrize("vis_chunks", vis_chunks)
def test_dask_conversion(input_schema, output_schema, vis_chunks):
    from africanus.model.coherency.dask import convert as da_convert

    vis_shape = tuple(sum(dim_chunks) for dim_chunks in vis_chunks)
    input_shape = np.asarray(input_schema).shape
    vis = visibility_factory(
        vis_shape, input_shape, backend="dask", vis_chunks=vis_chunks
    )

    da_vis = da_convert(vis, input_schema, output_schema)
    np_vis = np_convert(vis.compute(), input_schema, output_schema)
    assert np.all(da_vis == np_vis)
