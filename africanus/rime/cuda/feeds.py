# -*- coding: utf-8 -*-


import logging
from os.path import join as pjoin

import numpy as np

from africanus.rime.feeds import FEED_ROTATION_DOCS
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


def _key_fn(parallactic_angles, feed_type):
    return (parallactic_angles.dtype, feed_type)


@memoize_on_key(_key_fn)
def _generate_kernel(parallactic_angles, feed_type):
    dtype = parallactic_angles.dtype

    # Block sizes
    if dtype == np.float32:
        block = (1024, 1, 1)
    elif dtype == np.float64:
        block = (512, 1, 1)
    else:
        raise TypeError("Unhandled type %s" % dtype)

    # Create template
    render = jinja_env.get_template(_TEMPLATE_PATH).render
    name = "feed_rotation"

    code = render(kernel_name=name,
                  feed_type=feed_type,
                  sincos_fn=cuda_function('sincos', dtype),
                  pa_type=_get_typename(dtype),
                  out_type=_get_typename(dtype))

    code = code.encode('utf-8')

    # Complex output type
    out_dtype = np.result_type(dtype, np.complex64)
    return cp.RawKernel(code, name), block, out_dtype


@requires_optional("cupy", opt_import_error)
def feed_rotation(parallactic_angles, feed_type='linear'):
    """ Cupy implementation of the feed_rotation kernel. """
    kernel, block, out_dtype = _generate_kernel(parallactic_angles, feed_type)
    in_shape = parallactic_angles.shape
    parallactic_angles = parallactic_angles.ravel()
    grid = grids((parallactic_angles.shape[0], 1, 1), block)
    out = cp.empty(shape=(parallactic_angles.shape[0], 4), dtype=out_dtype)

    try:
        kernel(grid, block, (parallactic_angles, out))
    except CompileException:
        log.exception(format_code(kernel.code))
        raise

    return out.reshape(in_shape + (2, 2))


try:
    feed_rotation.__doc__ = FEED_ROTATION_DOCS.substitute(
                                array_type=":class:`cupy.ndarray`")
except AttributeError:
    pass
