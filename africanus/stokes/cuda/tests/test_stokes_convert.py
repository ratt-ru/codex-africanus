# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest


from africanus.stokes.stokes_conversion import (
                        stokes_convert as np_stokes_convert)
from africanus.stokes.cuda.stokes_conversion import stokes_convert
from africanus.stokes.tests.test_stokes import (stokes_corr_cases,
                                                stokes_corr_int_cases,
                                                visibility_factory)


@pytest.mark.skip
def test_stokes_schemas(in_type, input_schema,
                        out_type, output_schema,
                        vis_shape):
    input_shape = np.asarray(input_schema).shape
    output_shape = np.asarray(output_schema).shape

    vis = visibility_factory(vis_shape, input_shape, in_type)
    xformed_vis = np_stokes_convert(vis, input_schema, output_schema)
    assert xformed_vis.shape == vis_shape + output_shape


@pytest.mark.parametrize("in_type, input_schema, out_type, output_schema",
                         stokes_corr_cases)
@pytest.mark.parametrize("vis_shape", [
    (10, 5, 3),
    (6, 8),
    (15,),
])
def test_cuda_stokes_convert(in_type, input_schema,
                             out_type, output_schema,
                             vis_shape):
    cp = pytest.importorskip('cupy')

    input_shape = np.asarray(input_schema).shape
    output_shape = np.asarray(output_schema).shape
    vis = visibility_factory(vis_shape, input_shape, in_type)

    cp_out = stokes_convert(cp.asarray(vis), input_schema, output_schema)
    np_out = np_stokes_convert(vis, input_schema, output_schema)

    np.testing.assert_array_almost_equal(cp.asnumpy(cp_out), np_out)
