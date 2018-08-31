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
    (['I', 'Q'], ['XX', 'YY']),
])
@pytest.mark.parametrize("vis_shape", [
    (10, 5, 3),
    (6, 8),
    (15,),
])
def test_stokes_conversion(input_schema, output_schema, vis_shape):
    input_shape = np.asarray(input_schema).shape
    output_shape = np.asarray(output_schema).shape

    shape = vis_shape + input_shape

    vis = np.arange(1.0, np.product(shape) + 1.0)
    vis = vis.reshape(shape)
    vis = vis + vis*1j

    xformed_vis = convert(vis, input_schema, output_schema)
    assert xformed_vis.shape == vis_shape + output_shape
