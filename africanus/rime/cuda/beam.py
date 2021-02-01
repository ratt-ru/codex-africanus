# -*- coding: utf-8 -*-

from functools import reduce
import logging
from operator import mul
from pathlib import Path

import numpy as np

from africanus.rime.fast_beam_cubes import BEAM_CUBE_DOCS
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


_MAIN_TEMPLATE_PATH = Path("rime", "cuda", "beam.cu.j2")
_INTERP_TEMPLATE_PATH = Path("rime", "cuda", "beam_freq_interp.cu.j2")

BEAM_NUD_LIMIT = 128


def _freq_interp_key(beam_freq_map, frequencies):
    return (beam_freq_map.dtype, frequencies.dtype)


@memoize_on_key(_freq_interp_key)
def _generate_interp_kernel(beam_freq_map, frequencies):
    render = jinja_env.get_template(str(_INTERP_TEMPLATE_PATH)).render
    name = "beam_cube_freq_interp"

    block = (1024, 1, 1)

    code = render(kernel_name=name,
                  beam_nud_limit=BEAM_NUD_LIMIT,
                  blockdimx=block[0],
                  beam_freq_type=_get_typename(beam_freq_map.dtype),
                  freq_type=_get_typename(frequencies.dtype))

    code = code.encode('utf-8')
    dtype = np.result_type(beam_freq_map, frequencies)

    return cp.RawKernel(code, name), block, dtype


def _main_key_fn(beam, beam_lm_ext, beam_freq_map,
                 lm, parangles, pointing_errors,
                 antenna_scaling, frequencies,
                 dde_dims, ncorr):
    return (beam.dtype, beam.ndim, beam_lm_ext.dtype, beam_freq_map.dtype,
            lm.dtype, parangles.dtype, pointing_errors.dtype,
            antenna_scaling.dtype, frequencies.dtype, dde_dims, ncorr)


# Value to use in a bit shift to recover channel from flattened
# channel/correlation index
_corr_shifter = {4: 2, 2: 1, 1: 0}


@memoize_on_key(_main_key_fn)
def _generate_main_kernel(beam, beam_lm_ext, beam_freq_map,
                          lm, parangles, pointing_errors,
                          antenna_scaling, frequencies,
                          dde_dims, ncorr):

    beam_lw, beam_mh, beam_nud = beam.shape[:3]

    if beam_lw < 2 or beam_mh < 2 or beam_nud < 2:
        raise ValueError("(beam_lw, beam_mh, beam_nud) < 2 "
                         "to linearly interpolate")

    # Create template
    render = jinja_env.get_template(str(_MAIN_TEMPLATE_PATH)).render
    name = "beam_cube_dde"
    dtype = beam.dtype

    if dtype == np.complex64:
        block = (32, 32, 1)
    elif dtype == np.complex128:
        block = (32, 16, 1)
    else:
        raise TypeError("Need complex beam cube '%s'" % beam.dtype)

    try:
        corr_shift = _corr_shifter[ncorr]
    except KeyError:
        raise ValueError("Number of Correlations not in %s"
                         % list(_corr_shifter.keys()))

    coord_type = np.result_type(beam_lm_ext, lm, parangles,
                                pointing_errors, antenna_scaling,
                                np.float32)

    assert coord_type in (np.float32, np.float64)

    code = render(kernel_name=name,
                  blockdimx=block[0],
                  blockdimy=block[1],
                  blockdimz=block[2],
                  corr_shift=corr_shift,
                  ncorr=ncorr,
                  beam_nud_limit=BEAM_NUD_LIMIT,
                  # Beam type and manipulation functions
                  beam_type=_get_typename(beam.real.dtype),
                  beam_dims=beam.ndim,
                  make2_beam_fn=cuda_function('make2', beam.real.dtype),
                  beam_sqrt_fn=cuda_function('sqrt', beam.real.dtype),
                  beam_rsqrt_fn=cuda_function('rsqrt', beam.real.dtype),
                  # Coordinate type and manipulation functions
                  FT=_get_typename(coord_type),
                  floor_fn=cuda_function('floor', coord_type),
                  min_fn=cuda_function('min', coord_type),
                  max_fn=cuda_function('max', coord_type),
                  cos_fn=cuda_function('cos', coord_type),
                  sin_fn=cuda_function('sin', coord_type),
                  lm_ext_type=_get_typename(beam_lm_ext.dtype),
                  beam_freq_type=_get_typename(beam_freq_map.dtype),
                  lm_type=_get_typename(lm.dtype),
                  pa_type=_get_typename(parangles.dtype),
                  pe_type=_get_typename(pointing_errors.dtype),
                  as_type=_get_typename(antenna_scaling.dtype),
                  freq_type=_get_typename(frequencies.dtype),
                  dde_type=_get_typename(beam.real.dtype),
                  dde_dims=dde_dims)

    code = code.encode('utf-8')

    # Complex output type
    return cp.RawKernel(code, name), block, dtype


@requires_optional('cupy', opt_import_error)
def beam_cube_dde(beam, beam_lm_ext, beam_freq_map,
                  lm, parangles, pointing_errors,
                  antenna_scaling, frequencies):

    corrs = beam.shape[3:]

    if beam.shape[2] >= BEAM_NUD_LIMIT:
        raise ValueError("beam_nud exceeds %d" % BEAM_NUD_LIMIT)

    nsrc = lm.shape[0]
    ntime, na = parangles.shape
    nchan = frequencies.shape[0]
    ncorr = reduce(mul, corrs, 1)
    nchancorr = nchan*ncorr

    oshape = (nsrc, ntime, na, nchan) + corrs

    if len(corrs) > 1:
        # Flatten the beam correlation dims
        fbeam = beam.reshape(beam.shape[:3] + (ncorr,))
    else:
        fbeam = beam

    # Generate frequency interpolation kernel
    ikernel, iblock, idt = _generate_interp_kernel(beam_freq_map, frequencies)

    # Generate main beam cube kernel
    kernel, block, dtype = _generate_main_kernel(fbeam, beam_lm_ext,
                                                 beam_freq_map,
                                                 lm, parangles,
                                                 pointing_errors,
                                                 antenna_scaling,
                                                 frequencies,
                                                 len(oshape),
                                                 ncorr)
    # Call frequency interpolation kernel
    igrid = grids((nchan, 1, 1), iblock)
    freq_data = cp.empty((3, nchan), dtype=frequencies.dtype)

    try:
        ikernel(igrid, iblock, (frequencies, beam_freq_map, freq_data))
    except CompileException:
        log.exception(format_code(ikernel.code))
        raise

    # Call main beam cube kernel
    out = cp.empty((nsrc, ntime, na, nchan) + (ncorr,), dtype=beam.dtype)
    grid = grids((nchancorr, na, ntime), block)

    try:
        kernel(grid, block, (fbeam, beam_lm_ext, beam_freq_map,
                             lm, parangles, pointing_errors,
                             antenna_scaling, frequencies, freq_data,
                             nsrc, out))
    except CompileException:
        log.exception(format_code(kernel.code))
        raise

    return out.reshape(oshape)


try:
    beam_cube_dde.__doc__ = BEAM_CUBE_DOCS.substitute(
                                array_type=":class:`cupy.ndarray`")
except AttributeError:
    pass
