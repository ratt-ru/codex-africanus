# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from operator import mul
from os.path import join as pjoin

import numpy as np

from africanus.compatibility import reduce
from africanus.rime.predict import PREDICT_DOCS, predict_checks
from africanus.util.code import format_code, memoize_on_key
from africanus.util.cuda import cuda_function, cuda_type, grids
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
    """ Hash on array datatypes and rank """
    return tuple((a.dtype, a.ndim) if isinstance(a, (np.ndarray, cp.ndarray))
                 else a for a in args)


@memoize_on_key(_key_fn)
def _generate_kernel(time_index, antenna1, antenna2,
                     dde1_jones, source_coh, dde2_jones,
                     die1_jones, base_vis, die2_jones,
                     corrs, out_ndim):

    tup = predict_checks(time_index, antenna1, antenna2,
                         dde1_jones, source_coh, dde2_jones,
                         die1_jones, base_vis, die2_jones)

    (have_ddes1, have_coh, have_ddes2, have_dies1, have_bvis, have_dies2) = tup

    # Create template
    render = jinja_env.get_template(_TEMPLATE_PATH).render
    name = "predict_vis"

    # Complex output type
    out_dtype = np.result_type(dde1_jones, source_coh, dde2_jones,
                               die1_jones, base_vis, die2_jones)

    code = render(kernel_name=name,
                  have_dde1=have_ddes1,
                  dde1_type=cuda_type(dde1_jones) if have_ddes1 else "int",
                  dde1_ndim=dde1_jones.ndim if have_ddes1 else 1,
                  have_dde2=have_ddes2,
                  dde2_type=cuda_type(dde2_jones) if have_ddes2 else "int",
                  dde2_ndim=dde2_jones.ndim if have_ddes2 else 1,
                  have_coh=have_coh,
                  coh_type=cuda_type(source_coh) if have_coh else "int",
                  coh_ndim=source_coh.ndim if have_coh else 1,
                  have_die1=have_dies1,
                  die1_type=cuda_type(die1_jones) if have_dies1 else "int",
                  die1_ndim=die1_jones.ndim if have_dies1 else 1,
                  have_base_vis=have_bvis,
                  base_vis_type=cuda_type(base_vis) if have_bvis else "int",
                  base_vis_ndim=base_vis.ndim if have_bvis else 1,
                  have_die2=have_dies2,
                  die2_type=cuda_type(die2_jones) if have_dies2 else "int",
                  die2_ndim=die2_jones.ndim if have_dies2 else 1,
                  out_type=cuda_type(out_dtype),
                  corrs=reduce(mul, corrs, 1),
                  out_ndim=out_ndim)
    code = code.encode('utf-8')

    block = (512, 1, 1)

    return cp.RawKernel(code, name), block, out_dtype


@requires_optional("cupy", opt_import_error)
def predict_vis(time_index, antenna1, antenna2,
                dde1_jones, source_coh, dde2_jones,
                die1_jones, base_vis, die2_jones):
    """ Cupy implementation of the feed_rotation kernel. """

    have_ddes = dde1_jones is not None and dde2_jones is not None
    have_dies = die1_jones is not None and die2_jones is not None
    have_coh = source_coh is not None
    have_bvis = base_vis is not None

    # Infer the output shape
    if have_ddes:
        row = time_index.shape[0]
        chan = dde1_jones.shape[3]
        corrs = dde1_jones.shape[4:]
    elif have_coh:
        row = time_index.shape[0]
        chan = source_coh.shape[2]
        corrs = source_coh.shape[3:]
    elif have_dies:
        row = time_index.shape[0]
        chan = die1_jones.shape[2]
        corrs = die1_jones.shape[3:]
    else:
        raise ValueError("Insufficient inputs supplied for determining "
                         "the output shape")

    ncorrs = len(corrs)

    # Flatten correlations
    if ncorrs == 2:
        flat_corrs = reduce(mul, corrs, 1)

        if have_ddes:
            dde_shape = dde1_jones.shape[:-ncorrs] + (flat_corrs,)
            dde1_jones = dde1_jones.reshape(dde_shape)
            dde2_jones = dde2_jones.reshape(dde_shape)

        if have_coh:
            coh_shape = source_coh.shape[:-ncorrs] + (flat_corrs,)
            source_coh = source_coh.reshape(coh_shape)

        if have_dies:
            die_shape = die1_jones.shape[:-ncorrs] + (flat_corrs,)
            die1_jones = die1_jones.reshape(die_shape)
            die2_jones = die2_jones.reshape(die_shape)

        if have_bvis:
            bvis_shape = base_vis.shape[:-ncorrs] + (flat_corrs,)
            base_vis = base_vis.reshape(bvis_shape)

    elif ncorrs == 1:
        flat_corrs = corrs[0]
    else:
        raise ValueError("Invalid correlation setup %s" % (corrs,))

    out_shape = (row, chan) + (flat_corrs,)

    kernel, block, out_dtype = _generate_kernel(time_index,
                                                antenna1,
                                                antenna2,
                                                dde1_jones,
                                                source_coh,
                                                dde2_jones,
                                                die1_jones,
                                                base_vis,
                                                die2_jones,
                                                corrs,
                                                len(out_shape))

    grid = grids((1, 1, 1), block)
    out = cp.empty(shape=out_shape, dtype=out_dtype)

    print(format_code(kernel.code))

    args = (time_index, antenna1, antenna2,
            dde1_jones, source_coh, dde2_jones,
            die1_jones, base_vis, die2_jones,
            out)

    try:
        kernel(grid, block, tuple(a for a in args if a is not None))
    except CompileException:
        log.exception(format_code(kernel.code))
        raise

    return out.reshape((row, chan) + corrs)


try:
    predict_vis.__doc__ = PREDICT_DOCS.substitute(
                                array_type=":class:`cupy.ndarray`",
                                extra_notes="")
except AttributeError:
    pass
