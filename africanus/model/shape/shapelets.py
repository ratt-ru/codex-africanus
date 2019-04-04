import logging
from os.path import join as pjoin

import numpy as np

from africanus.util.jinja2 import jinja_env
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

def _key_fn(uvw, coeffs, beta):
    return (uvw.dtype, coeffs.dtype, beta.dtype)

_TEMPLATE_PATH = pjoin("model", "shape", "cuda", "shapelets.cu.j2")

@memoize_on_key(_key_fn)
def _generate_kernel(uvw, coeffs, beta):
    out_dtype = np.result_type(uvw, coeffs, beta)
    
    blockdimx = 32
    blockdimy = 32
    block = (blockdimx, blockdimy, 0)

    render = jinja_env.get_template(_TEMPLATE_PATH).render
    name = "shapelet"

    code = render(kernel_name=name,
        uvw_type=_get_typename(uvw.dtype),
        coeffs_type=_get_typename(coeffs.dtype),
        beta_type=_get_typename(beta.dtype),
        out_type=_get_typename(out_dtype),
        blockdimx=blockdimx,
        blockdimy=blockdimy).encode('utf-8')

    out_dtype = np.result_type(out_dtype, np.complex64)
    return cp.RawKernel(code, name), block, out_dtype

@requires_optional("cupy")
def shapelet(uvw, coeffs, beta):
    kernel, block, out_dtype = _generate_kernel(uvw, coeffs, beta)
    grid = grids((uvw.shape[0], uvw.shape[1], 1), block)
    out = cp.empty(shape=(uvw.shape[0], uvw.shape[1]), dtype=out_dtype)

    try:
        kernel(grid, block, (uvw, coeffs, beta, out))
    except CompileException:
        log.exception(format_code(kernel.code))
        raise
    return out