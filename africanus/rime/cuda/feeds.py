# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from os.path import join as pjoin

import numpy as np

from africanus.constants import minus_two_pi_over_c
from africanus.util.code import format_code, memoize_on_key
from africanus.util.cuda import cuda_function, grids
from africanus.util.jinja2 import jinja_env
from africanus.util.requirements import requires_optional

try:
    import cupy as cp
    from cupy.core._scalar import get_typename as _get_typename
    from cupy.cuda.compiler import CompileException
except ImportError as e:
    opt_import_error = e
else:
    opt_import_error = None


log = logging.getLogger(__name__)


_TEMPLATE_PATH = pjoin("rime", "cuda", "feeds.cu.j2")


def _key_fn(parallactic_angles):
    return parallactic_angles.dtype


@memoize_on_key(_key_fn)
def _generate_kernel(parallactic_angles):
    # Floating point output type

    # Block sizes
    blockdimx = 512
    block = (blockdimx, 1, 1)

    # Create template
    render = jinja_env.get_template(_TEMPLATE_PATH).render
    name = "phase_delay"

    code = render(kernel_name=name,
                  pa_type=_get_typename(parallactic_angles.dtype),
                  out_type=_get_typename(parallactic_angles.dtype))

    code = code.encode('utf-8')

    # Complex output type
    out_dtype = np.result_type(parallactic_angles.dtype, np.complex64)
    return cp.RawKernel(code, name), block, code, out_dtype


@requires_optional("cupy", opt_import_error)
def feed_rotation(parallactic_angles):
    """
    Cupy implementation of the feed_rotation kernel.

    TODO(sjperkins). Fill in the documentation with the numba doc template
    """
    kernel, block, code, out_dtype = _generate_kernel(parallactic_angles)
    in_shape = parallactic_angles.shape
    parallactic_angles = parallactic_angles.ravel()
    grid = grids((parallactic_angles.shape[0], 1, 1), block)
    out = cp.empty(shape=(parallactic_angles.shape[0], 4),
                   dtype=out_dtype)

    try:
        kernel(grid, block, (parallactic_angles,))
    except CompileException:
        log.exception(format_kernel(code))
        raise

    return out.reshape(shape + (2, 2))
