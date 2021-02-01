# -*- coding: utf-8 -*-


import logging
from os.path import join as pjoin

import numpy as np

from africanus.constants import minus_two_pi_over_c
from africanus.util.jinja2 import jinja_env
from africanus.rime.phase import PHASE_DELAY_DOCS
from africanus.util.code import memoize_on_key, format_code
from africanus.util.cuda import cuda_function, grids
from africanus.util.requirements import requires_optional

try:
    import cupy as cp
    from cupy.core._scalar import get_typename as _get_typename
    from cupy.cuda.compiler import CompileException
except ImportError:
    pass

log = logging.getLogger(__name__)


def _key_fn(lm, uvw, frequency):
    return (lm.dtype, uvw.dtype, frequency.dtype)


_TEMPLATE_PATH = pjoin("rime", "cuda", "phase.cu.j2")


@memoize_on_key(_key_fn)
def _generate_kernel(lm, uvw, frequency):
    # Floating point output type
    out_dtype = np.result_type(lm, uvw, frequency)

    # Block sizes
    blockdimx = 32 if frequency.dtype == np.float32 else 16
    blockdimy = 32 if uvw.dtype == np.float32 else 16
    block = (blockdimx, blockdimy, 1)

    # Create template
    render = jinja_env.get_template(_TEMPLATE_PATH).render
    name = "phase_delay"

    code = render(kernel_name=name,
                  lm_type=_get_typename(lm.dtype),
                  uvw_type=_get_typename(uvw.dtype),
                  freq_type=_get_typename(frequency.dtype),
                  out_type=_get_typename(out_dtype),
                  sqrt_fn=cuda_function('sqrt', lm.dtype),
                  sincos_fn=cuda_function('sincos', out_dtype),
                  minus_two_pi_over_c=minus_two_pi_over_c,
                  blockdimx=blockdimx,
                  blockdimy=blockdimy).encode('utf-8')

    # Complex output type
    out_dtype = np.result_type(out_dtype, np.complex64)
    return cp.RawKernel(code, name), block, out_dtype


@requires_optional("cupy")
def phase_delay(lm, uvw, frequency):
    kernel, block, out_dtype = _generate_kernel(lm, uvw, frequency)
    grid = grids((frequency.shape[0], uvw.shape[0], 1), block)
    out = cp.empty(shape=(lm.shape[0], uvw.shape[0], frequency.shape[0]),
                   dtype=out_dtype)

    try:
        kernel(grid, block, (lm, uvw, frequency, out))
    except CompileException:
        log.exception(format_code(kernel.code))
        raise

    return out


try:
    phase_delay.__doc__ = PHASE_DELAY_DOCS.substitute(
                                array_type=':class:`cupy.ndarray`')
except AttributeError:
    pass
