#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

import numpy as np
from numpy.testing import assert_array_almost_equal

import pytest


def rf(*a, **kw):
    return np.random.random(*a, **kw)


def rc(*a, **kw):
    return rf(*a, **kw) + 1j*rf(*a, **kw)


chunk_parametrization = pytest.mark.parametrize("chunks", [
    {
        'source':  (2, 3, 4, 2, 2, 2, 2, 2, 2),
        'time': (2, 1, 1),
        'rows': (4, 4, 2),
        'antenna': (4,),
        'channels': (3, 2),
    }])

corr_shape_parametrization = pytest.mark.parametrize(
    'corr_shape, idm, einsum_sig1, einsum_sig2', [
        ((1,), (1,), "srci,srci,srci->rci", "rci,rci,rci->rci"),
        ((2,), (1, 1), "srci,srci,srci->rci", "rci,rci,rci->rci"),
        ((2, 2), ((1, 0), (0, 1)),
            "srcij,srcjk,srclk->rcil", "rcij,rcjk,rclk->rcil")
    ])


dde_presence_parametrization = pytest.mark.parametrize('a1j,blj,a2j', [
    [True, True, True],
    [True, False, True],
    [False, True, False],
])

die_presence_parametrization = pytest.mark.parametrize('g1j,bvis,g2j', [
    [True, True, True],
    [True, False, True],
    [False, True, False],
])


@corr_shape_parametrization
@dde_presence_parametrization
@die_presence_parametrization
@chunk_parametrization
def test_predict_vis(corr_shape, idm, einsum_sig1, einsum_sig2,
                     a1j, blj, a2j, g1j, bvis, g2j,
                     chunks):
    from africanus.rime.predict import predict_vis

    s = sum(chunks['source'])
    t = sum(chunks['time'])
    a = sum(chunks['antenna'])
    c = sum(chunks['channels'])
    r = sum(chunks['rows'])

    a1_jones = rc((s, t, a, c) + corr_shape)
    bl_jones = rc((s, r, c) + corr_shape)
    a2_jones = rc((s, t, a, c) + corr_shape)
    g1_jones = rc((t, a, c) + corr_shape)
    base_vis = rc((r, c) + corr_shape)
    g2_jones = rc((t, a, c) + corr_shape)

    #  Row indices into the above time/ant indexed arrays
    time_idx = np.asarray([0, 0, 1, 1, 2, 2, 2, 2, 3, 3])
    ant1 = np.asarray([0, 0, 0, 0, 1, 1, 1, 2, 2, 3])
    ant2 = np.asarray([0, 1, 2, 3, 1, 2, 3, 2, 3, 3])

    assert ant1.size == r

    model_vis = predict_vis(time_idx, ant1, ant2,
                            a1_jones if a1j else None,
                            bl_jones if blj else None,
                            a2_jones if a2j else None,
                            g1_jones if g1j else None,
                            base_vis if bvis else None,
                            g2_jones if g2j else None)

    assert model_vis.shape == (r, c) + corr_shape

    def _id(array):
        return np.broadcast_to(idm, array.shape)

    # For einsum, convert (time, ant) dimensions to row
    # or ID matrices if the input is present
    a1_jones = a1_jones[:, time_idx, ant1] if a1j else _id(bl_jones)
    bl_jones = bl_jones if blj else _id(bl_jones)
    a2_jones = a2_jones[:, time_idx, ant2].conj() if a2j else _id(bl_jones)

    v = np.einsum(einsum_sig1, a1_jones, bl_jones, a2_jones)

    if bvis:
        v += base_vis

    # Convert (time, ant) dimensions to row or
    # or ID matrices if input is not present
    g1_jones = g1_jones[time_idx, ant1] if g1j else _id(v)
    g2_jones = g2_jones[time_idx, ant2].conj() if g2j else _id(v)

    v = np.einsum(einsum_sig2, g1_jones, v, g2_jones)

    assert_array_almost_equal(v, model_vis)


@corr_shape_parametrization
@dde_presence_parametrization
@die_presence_parametrization
@chunk_parametrization
def test_dask_predict_vis(corr_shape, idm, einsum_sig1, einsum_sig2,
                          a1j, blj, a2j, g1j, bvis, g2j,
                          chunks):

    da = pytest.importorskip('dask.array')
    import numpy as np
    import dask

    from africanus.rime.predict import predict_vis as np_predict_vis
    from africanus.rime.dask import predict_vis

    # chunk sizes
    sc = chunks['source']
    tc = chunks['time']
    rrc = chunks['rows']
    ac = chunks['antenna']
    cc = chunks['channels']

    # dimension sizes
    s = sum(sc)       # sources
    t = sum(tc)       # times
    a = sum(ac)       # antennas
    c = sum(cc)       # channels
    r = sum(rrc)      # rows

    a1_jones = rc((s, t, a, c) + corr_shape)
    a2_jones = rc((s, t, a, c) + corr_shape)
    bl_jones = rc((s, r, c) + corr_shape)
    g1_jones = rc((t, a, c) + corr_shape)
    base_vis = rc((r, c) + corr_shape)
    g2_jones = rc((t, a, c) + corr_shape)

    #  Row indices into the above time/ant indexed arrays
    time_idx = np.asarray([0, 0, 1, 1, 2, 2, 2, 2, 3, 3])
    ant1 = np.asarray([0, 0, 0, 0, 1, 1, 1, 2, 2, 3])
    ant2 = np.asarray([0, 1, 2, 3, 1, 2, 3, 2, 3, 3])

    assert ant1.size == r

    np_model_vis = np_predict_vis(time_idx, ant1, ant2,
                                  a1_jones if a1j else None,
                                  bl_jones if blj else None,
                                  a2_jones if a2j else None,
                                  g1_jones if g1j else None,
                                  base_vis if bvis else None,
                                  g2_jones if g2j else None)

    da_time_idx = da.from_array(time_idx, chunks=rrc)
    da_ant1 = da.from_array(ant1, chunks=rrc)
    da_ant2 = da.from_array(ant2, chunks=rrc)

    da_a1_jones = da.from_array(a1_jones, chunks=(sc, tc, ac, cc) + corr_shape)
    da_bl_jones = da.from_array(bl_jones, chunks=(sc, rrc, cc) + corr_shape)
    da_a2_jones = da.from_array(a2_jones, chunks=(sc, tc, ac, cc) + corr_shape)
    da_g1_jones = da.from_array(g1_jones, chunks=(tc, ac, cc) + corr_shape)
    da_base_vis = da.from_array(base_vis, chunks=(rrc, cc) + corr_shape)
    da_g2_jones = da.from_array(g2_jones, chunks=(tc, ac, cc) + corr_shape)

    args = (da_time_idx, da_ant1, da_ant2,
            da_a1_jones if a1j else None,
            da_bl_jones if blj else None,
            da_a2_jones if a2j else None,
            da_g1_jones if g1j else None,
            da_base_vis if bvis else None,
            da_g2_jones if g2j else None)

    stream_model_vis = predict_vis(*args, streams=True)
    fan_model_vis = predict_vis(*args, streams=False)

    stream_model_vis, fan_model_vis = dask.compute(stream_model_vis,
                                                   fan_model_vis)

    assert_array_almost_equal(fan_model_vis, np_model_vis)
    assert_array_almost_equal(stream_model_vis, fan_model_vis)
