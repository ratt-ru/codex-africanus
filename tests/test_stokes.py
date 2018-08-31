#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

import numpy as np

import pytest


from africanus.stokes.transformation_defs import convert


@pytest.mark.parametrize("input_schema, output_schema", [
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
])
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

    xformed_vis = convert(vis, input_schema, output_schema)
    assert xformed_vis.shape == vis_shape + output_shape


def test_stokes_conversion():
    I, Q, U, V = [1.0, 2.0, 3.0, 4.0]

    # Check conversion to linear
    vis = convert(np.asarray([[I, Q, U, V]]),
                  ['I', 'Q', 'U', 'V'],
                  ['XX', 'XY', 'YX', 'YY'])

    XX, XY, YX, YY = vis[0]
    assert np.all(vis == [[I + Q, U + V*1j, U - V*1j, I - Q]])

    # Check conversion to circular
    vis = convert(np.asarray([[I, Q, U, V]]),
                  ['I', 'Q', 'U', 'V'],
                  ['RR', 'RL', 'LR', 'LL'])

    RR, RL, LR, LL = vis[0]
    assert np.all(vis == [[I + V, Q + U*1j, Q - U*1j, I - V]])

    # linear to stokes
    stokes = convert(np.asarray([[XX, XY, YX, YY]]),
                     ['XX', 'XY', 'YX', 'YY'],
                     ['I', 'Q', 'U', 'V'])

    assert np.all(stokes == [[I, Q, U, V]])

    # circular to stokes
    stokes = convert(np.asarray([[RR, RL, LR, LL]]),
                     ['RR', 'RL', 'LR', 'LL'],
                     ['I', 'Q', 'U', 'V'])

    assert np.all(stokes == [[I, Q, U, V]])
