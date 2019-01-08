# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from os.path import join as pjoin

import numpy as np

from africanus.rime.predict import PREDICT_DOCS
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


_TEMPLATE_PATH = pjoin("rime", "cuda", "predict.cu.j2")


def _key_fn(*args):
    return tuple(a.dtype for a in args if a is not None)


@memoize_on_key(_key_fn)
def _generate_kernel(*args):

    # Create template
    render = jinja_env.get_template(_TEMPLATE_PATH).render
    name = "predict_vis"

    code = render(kernel_name=name)
    code = code.encode('utf-8')

    block = (512, 1, 1)

    # Complex output type
    out_dtype = np.result_type(*args)
    return cp.RawKernel(code, name), block, out_dtype


@requires_optional("cupy", opt_import_error)
def predict_vis(time_index, antenna1, antenna2,
                dde1_jones, source_coh, dde2_jones,
                die1_jones, base_vis, die2_jones):
    """ Cupy implementation of the feed_rotation kernel. """
    kernel, block, out_dtype = _generate_kernel(time_index,
                                                antenna1,
                                                antenna2,
                                                dde1_jones,
                                                source_coh,
                                                dde2_jones,
                                                die1_jones,
                                                base_vis,
                                                die2_jones)

    grid = grids((1, 1, 1), block)
    out = cp.empty(shape=(1, 4), dtype=out_dtype)

    try:
        kernel(grid, block, (time_index, antenna1, antenna2))
    except CompileException:
        log.exception(format_code(kernel.code))
        raise

    return out


try:
    predict_vis.__doc__ = PREDICT_DOCS.substitute(
                                array_type=":class:`cupy.ndarray`",
                                extra_notes="")
except AttributeError:
    pass
