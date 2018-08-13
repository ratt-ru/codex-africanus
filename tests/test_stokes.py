#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

import numpy as np

import pytest


from africanus.stokes.transformation_defs import transformer


@pytest.mark.parametrize("in_outs", [
    ([['XX'], ['YY']], ['I', 'Q']),
    (['XX', 'YY'], ['I', 'Q']),
    (['XX', 'XY', 'YX', 'YY'], ['I', 'Q', 'U', 'V']),
    ([['XX', 'XY'], ['YX', 'YY']], [['I', 'Q'], ['U', 'V']]),
    (['I', 'Q', 'U', 'V'], ['XX', 'XY', 'YX', 'YY']),
    ([['I', 'Q'], ['U', 'V']], [['XX', 'XY'], ['YX', 'YY']]),
    (['I', 'Q'], ['XX', 'YY']),
])
def test_stokes_conversion(in_outs):
    inputs, outputs = in_outs
    xform = transformer(inputs, outputs)

    input_shape = np.asarray(inputs).shape
    output_shape = np.asarray(outputs).shape

    nvis = 10
    vis_shape = (nvis,) + input_shape

    vis = np.arange(1.0, np.product(vis_shape) + 1.0)
    vis = vis.reshape(vis_shape)
    vis = vis + vis*1j

    xformed_vis = xform(vis)
    assert xformed_vis.shape == (nvis,) + output_shape
