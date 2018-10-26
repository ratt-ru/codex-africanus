#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

import numba
import numpy as np

import pytest


def rf(*a, **kw):
    return np.random.random(*a, **kw)


def rc(*a, **kw):
    return rf(*a, **kw) + 1j*rf(*a, **kw)


@pytest.mark.parametrize('corr_shape, idm, einsum_sig1, einsum_sig2', [
    ((1,), (1,), "srci,srci,srci->rci", "rci,rci,rci->rci"),
    ((2,), (1, 1), "srci,srci,srci->rci", "rci,rci,rci->rci"),
    ((2, 2), ((1,0),(0,1)), "srcij,srcjk,srckl->rcil", "rcij,rcjk,rckl->rcil"),
])
@pytest.mark.parametrize('a1j,blj,a2j', [
    [True, True, True],
    [True, False, True],
    [False, True, False],
])
@pytest.mark.parametrize('g1j,g2j',[
    [True, True],
    [False, False],
])

def test_predict_vis(corr_shape, idm, einsum_sig1, einsum_sig2,
                     a1j, blj, a2j, g1j, g2j):
    import numpy as np
    from africanus.rime.predict2 import predict_vis

    s = 2       # sources
    t = 4       # times
    a = 4       # antennas
    c = 5       # channels
    r = 10      # rows

    a1_jones = rc((s, t, a, c) + corr_shape)
    a2_jones = rc((s, t, a, c) + corr_shape)
    bl_jones = rc((s, r, c) + corr_shape)
    g1_jones = rc((t, a, c) + corr_shape)
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
                            g2_jones if g2j else None)

    assert model_vis.shape == (r, c) + corr_shape

    def _id(array):
        return np.broadcast_to(idm, array.shape)

    # For einsum, convert (time, ant) dimensions to row
    # or ID matrices if the input is not present
    a1_jones = a1_jones[:, time_idx, ant1] if a1j else _id(bl_jones)
    bl_jones = bl_jones if blj else _id(bl_jones)
    a2_jones = a2_jones[:, time_idx, ant2].conj() if a2j else _id(bl_jones)

    v = np.einsum(einsum_sig1, a1_jones, bl_jones, a2_jones)

    g1_jones = g1_jones[time_idx, ant1] if g1j else _id(v)
    g2_jones = g2_jones[time_idx, ant2].conj() if g2j else _id(v)

    v = np.einsum(einsum_sig2, g1_jones, v, g2_jones)

    assert np.allclose(v, model_vis)
