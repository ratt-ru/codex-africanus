# -*- coding: utf-8 -*-


import numpy as np
import pytest

from africanus.model.coherency import convert as np_convert
from africanus.model.coherency.cuda import convert
from africanus.model.coherency.tests.test_convert import (
                                                stokes_corr_cases,
                                                stokes_corr_int_cases,
                                                visibility_factory,
                                                vis_shape)


@pytest.mark.skip
def test_stokes_schemas(in_type, input_schema,
                        out_type, output_schema,
                        vis_shape):
    input_shape = np.asarray(input_schema).shape
    output_shape = np.asarray(output_schema).shape

    vis = visibility_factory(vis_shape, input_shape, in_type)
    xformed_vis = np_convert(vis, input_schema, output_schema)
    assert xformed_vis.shape == vis_shape + output_shape


@pytest.mark.parametrize("in_type, input_schema, out_type, output_schema",
                         stokes_corr_cases + stokes_corr_int_cases)
@pytest.mark.parametrize("vis_shape", vis_shape)
def test_cuda_convert(in_type, input_schema,
                      out_type, output_schema,
                      vis_shape):
    cp = pytest.importorskip('cupy')

    input_shape = np.asarray(input_schema).shape
    vis = visibility_factory(vis_shape, input_shape, in_type)

    cp_out = convert(cp.asarray(vis), input_schema, output_schema)
    np_out = np_convert(vis, input_schema, output_schema)

    np.testing.assert_array_almost_equal(cp.asnumpy(cp_out), np_out)
