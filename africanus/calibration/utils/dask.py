# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import wraps

from africanus.calibration.utils import CORRECT_VIS_DOCS, CORRUPT_VIS_DOCS
from africanus.calibration.utils import RESIDUAL_VIS_DOCS
from africanus.calibration.utils import correct_vis as np_correct_vis
from africanus.calibration.utils import corrupt_vis as np_corrupt_vis
from africanus.calibration.utils import residual_vis as np_residual_vis
from africanus.calibration.utils import check_type

try:
    from dask.array.core import blockwise
except ImportError as e:
    dask_import_error = e
else:
    dask_import_error = None

DIAG_DIAG = 0
DIAG = 1
FULL = 2

@wraps(np_corrupt_vis)
def _corrupt_vis_wrapper(time_bin_indices, time_bin_counts, antenna1,
                         antenna2, jones, model):
    return np_corrupt_vis(time_bin_indices, time_bin_counts, antenna1,
                          antenna2, jones[0][0], model[0])


@requires_optional('dask.array', dask_import_error)
def corrupt_vis(time_bin_indices, time_bin_counts, antenna1,
                antenna2, jones, model):

    mode = check_type(jones, model, vis_type='model')

    if mode == DIAG_DIAG:
        out_shape = ("row", "chan", "corr1")
        model_shape = ("row", "chan", "dir", "corr1")
        jones_shape = ("row", "ant", "chan", "dir", "corr1")
    elif mode == DIAG:
        out_shape = ("row", "chan", "corr1", "corr2")
        model_shape = ("row", "chan", "dir", "corr1", "corr2")
        jones_shape = ("row", "ant", "chan", "dir", "corr1")
    elif mode == FULL:
        out_shape = ("row", "chan", "corr1", "corr2")
        model_shape = ("row", "chan", "dir", "corr1", "corr2")
        jones_shape = ("row", "ant", "chan", "dir", "corr1", "corr2")
    else:
        raise ValueError("Unknown mode argument of %s" % mode)

    return blockwise(_corrupt_vis_wrapper, out_shape,
                     time_bin_indices, ("row",),
                     time_bin_counts, ("row",),
                     antenna1, ("row",),
                     antenna2, ("row",),
                     jones, jones_shape,
                     model, model_shape,
                     adjust_chunks={"row": antenna1.chunks[0]},
                     new_axes={"corr2": 2},  # why?
                     dtype=model.dtype,
                     align_arrays=False)


@wraps(np_correct_vis)
def _correct_vis_wrapper(time_bin_indices, time_bin_counts, antenna1,
                         antenna2, jones, vis, flag):
    return np_correct_vis(time_bin_indices, time_bin_counts, antenna1,
                          antenna2, jones[0][0], vis, flag)


@requires_optional('dask.array', dask_import_error)
def correct_vis(time_bin_indices, time_bin_counts, antenna1,
                antenna2, jones, vis, flag):

    mode = check_type(jones, vis)

    if mode == DIAG_DIAG:
        out_shape = ("row", "chan", "corr1")
        jones_shape = ("row", "ant", "chan", "dir", "corr1")
    elif mode == DIAG:
        out_shape = ("row", "chan", "corr1", "corr2")
        jones_shape = ("row", "ant", "chan", "dir", "corr1")
    elif mode == FULL:
        out_shape = ("row", "chan", "corr1", "corr2")
        jones_shape = ("row", "ant", "chan", "dir", "corr1", "corr2")
    else:
        raise ValueError("Unknown mode argument of %s" % mode)

    return blockwise(_correct_vis_wrapper, out_shape,
                     time_bin_indices, ("row",),
                     time_bin_counts, ("row",),
                     antenna1, ("row",),
                     antenna2, ("row",),
                     jones, jones_shape,
                     vis, out_shape,
                     flag, out_shape,
                     adjust_chunks={"row": antenna1.chunks[0]},
                     new_axes={"corr2": 2},  # why?
                     dtype=vis.dtype,
                     align_arrays=False)


@wraps(np_residual_vis)
def _residual_vis_wrapper(time_bin_indices, time_bin_counts, antenna1,
                          antenna2, jones, vis, flag, model):
    return np_residual_vis(time_bin_indices, time_bin_counts, antenna1,
                           antenna2, jones[0][0], vis, flag, model[0])


@requires_optional('dask.array', dask_import_error)
def residual_vis(time_bin_indices, time_bin_counts, antenna1,
                 antenna2, jones, vis, flag, model):

    mode = check_type(jones, vis)

    if mode == DIAG_DIAG:
        out_shape = ("row", "chan", "corr1")
        model_shape = ("row", "chan", "dir", "corr1")
        jones_shape = ("row", "ant", "chan", "dir", "corr1")
    elif mode == DIAG:
        out_shape = ("row", "chan", "corr1", "corr2")
        model_shape = ("row", "chan", "dir", "corr1", "corr2")
        jones_shape = ("row", "ant", "chan", "dir", "corr1")
    elif mode == FULL:
        out_shape = ("row", "chan", "corr1", "corr2")
        model_shape = ("row", "chan", "dir", "corr1", "corr2")
        jones_shape = ("row", "ant", "chan", "dir", "corr1", "corr2")
    else:
        raise ValueError("Unknown mode argument of %s" % mode)
    return blockwise(_residual_vis_wrapper, out_shape,
                     time_bin_indices, ("row",),
                     time_bin_counts, ("row",),
                     antenna1, ("row",),
                     antenna2, ("row",),
                     jones, jones_shape,
                     vis, out_shape,
                     flag, out_shape,
                     model, model_shape,
                     adjust_chunks={"row": antenna1.chunks[0]},
                     new_axes={"corr2": 2},  # why?
                     dtype=vis.dtype,
                     align_arrays=False)

corrupt_vis.__doc__ = CORRUPT_VIS_DOCS.substitute(
                        array_type=":class:`dask.array.Array`")

correct_vis.__doc__ = CORRECT_VIS_DOCS.substitute(
                        array_type=":class:`dask.array.Array`")

residual_vis.__doc__ = RESIDUAL_VIS_DOCS.substitute(
                        array_type=":class:`dask.array.Array`")