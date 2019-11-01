# -*- coding: utf-8 -*-

from africanus.calibration.utils.correct_vis import CORRECT_VIS_DOCS
from africanus.calibration.utils.corrupt_vis import CORRUPT_VIS_DOCS
from africanus.calibration.utils.residual_vis import RESIDUAL_VIS_DOCS
from africanus.calibration.utils.compute_and_corrupt_vis import (
                                COMPUTE_AND_CORRUPT_VIS_DOCS)
from africanus.calibration.utils import correct_vis as np_correct_vis
from africanus.calibration.utils import (compute_and_corrupt_vis as
                                         np_compute_and_corrupt_vis)
from africanus.calibration.utils import corrupt_vis as np_corrupt_vis
from africanus.calibration.utils import residual_vis as np_residual_vis
from africanus.calibration.utils import check_type
from africanus.calibration.utils.utils import DIAG_DIAG, DIAG, FULL
from africanus.util.requirements import requires_optional

try:
    from dask.array.core import blockwise
except ImportError as e:
    dask_import_error = e
else:
    dask_import_error = None


def _corrupt_vis_wrapper(time_bin_indices, time_bin_counts, antenna1,
                         antenna2, jones, model):
    return np_corrupt_vis(time_bin_indices, time_bin_counts, antenna1,
                          antenna2, jones[0][0], model[0])


@requires_optional('dask.array', dask_import_error)
def corrupt_vis(time_bin_indices, time_bin_counts, antenna1,
                antenna2, jones, model):

    mode = check_type(jones, model, vis_type='model')

    if jones.chunks[1][0] != jones.shape[1]:
        raise ValueError("Cannot chunk jones over antenna")
    if jones.chunks[3][0] != jones.shape[3]:
        raise ValueError("Cannot chunk jones over direction")
    if model.chunks[2][0] != model.shape[2]:
        raise ValueError("Cannot chunk model over direction")

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

    # the new_axes={"corr2": 2} is required because of a dask bug
    # see https://github.com/dask/dask/issues/5550
    return blockwise(_corrupt_vis_wrapper, out_shape,
                     time_bin_indices, ("row",),
                     time_bin_counts, ("row",),
                     antenna1, ("row",),
                     antenna2, ("row",),
                     jones, jones_shape,
                     model, model_shape,
                     adjust_chunks={"row": antenna1.chunks[0]},
                     new_axes={"corr2": 2},
                     dtype=model.dtype,
                     align_arrays=False)


def _compute_and_corrupt_vis_wrapper(time_bin_indices, time_bin_counts,
                                     antenna1, antenna2, jones, model,
                                     uvw, freq, lm):
    return np_compute_and_corrupt_vis(time_bin_indices, time_bin_counts,
                                      antenna1, antenna2, jones[0][0],
                                      model[0], uvw[0], freq, lm[0][0])


@requires_optional('dask.array', dask_import_error)
def compute_and_corrupt_vis(time_bin_indices, time_bin_counts,
                            antenna1, antenna2, jones, model,
                            uvw, freq, lm):

    if jones.chunks[1][0] != jones.shape[1]:
        raise ValueError("Cannot chunk jones over antenna")
    if jones.chunks[3][0] != jones.shape[3]:
        raise ValueError("Cannot chunk jones over direction")
    if model.chunks[2][0] != model.shape[2]:
        raise ValueError("Cannot chunk model over direction")
    if uvw.chunks[1][0] != uvw.shape[1]:
        raise ValueError("Cannot chunk uvw over last axis")
    if lm.chunks[1][0] != lm.shape[1]:
        raise ValueError("Cannot chunks lm over direction")
    if lm.chunks[2][0] != lm.shape[2]:
        raise ValueError("Cannot chunks lm over last axis")

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

    # the new_axes={"corr2": 2} is required because of a dask bug
    # see https://github.com/dask/dask/issues/5550
    return blockwise(_compute_and_corrupt_vis_wrapper, out_shape,
                     time_bin_indices, ("row",),
                     time_bin_counts, ("row",),
                     antenna1, ("row",),
                     antenna2, ("row",),
                     jones, jones_shape,
                     model, model_shape,
                     uvw, ("row", "three"),
                     freq, ("chan",),
                     lm, ("row", "dir", "two"),
                     adjust_chunks={"row": antenna1.chunks[0]},
                     new_axes={"corr2": 2},
                     dtype=model.dtype,
                     align_arrays=False)


def _correct_vis_wrapper(time_bin_indices, time_bin_counts, antenna1,
                         antenna2, jones, vis, flag):
    return np_correct_vis(time_bin_indices, time_bin_counts, antenna1,
                          antenna2, jones[0][0], vis, flag)


@requires_optional('dask.array', dask_import_error)
def correct_vis(time_bin_indices, time_bin_counts, antenna1,
                antenna2, jones, vis, flag):

    if jones.chunks[1][0] != jones.shape[1]:
        raise ValueError("Cannot chunk jones over antenna")
    if jones.chunks[3][0] != jones.shape[3]:
        raise ValueError("Cannot chunk jones over direction")

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

    # the new_axes={"corr2": 2} is required because of a dask bug
    # see https://github.com/dask/dask/issues/5550
    return blockwise(_correct_vis_wrapper, out_shape,
                     time_bin_indices, ("row",),
                     time_bin_counts, ("row",),
                     antenna1, ("row",),
                     antenna2, ("row",),
                     jones, jones_shape,
                     vis, out_shape,
                     flag, out_shape,
                     adjust_chunks={"row": antenna1.chunks[0]},
                     new_axes={"corr2": 2},
                     dtype=vis.dtype,
                     align_arrays=False)


def _residual_vis_wrapper(time_bin_indices, time_bin_counts, antenna1,
                          antenna2, jones, vis, flag, model):
    return np_residual_vis(time_bin_indices, time_bin_counts, antenna1,
                           antenna2, jones[0][0], vis, flag, model[0])


@requires_optional('dask.array', dask_import_error)
def residual_vis(time_bin_indices, time_bin_counts, antenna1,
                 antenna2, jones, vis, flag, model):

    if jones.chunks[1][0] != jones.shape[1]:
        raise ValueError("Cannot chunk jones over antenna")
    if jones.chunks[3][0] != jones.shape[3]:
        raise ValueError("Cannot chunk jones over direction")
    if model.chunks[2][0] != model.shape[2]:
        raise ValueError("Cannot chunk model over direction")

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

    # the new_axes={"corr2": 2} is required because of a dask bug
    # see https://github.com/dask/dask/issues/5550
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
                     new_axes={"corr2": 2},
                     dtype=vis.dtype,
                     align_arrays=False)


compute_and_corrupt_vis.__doc__ = COMPUTE_AND_CORRUPT_VIS_DOCS.substitute(
                                        array_type=":class:`dask.array.Array`")

corrupt_vis.__doc__ = CORRUPT_VIS_DOCS.substitute(
                        array_type=":class:`dask.array.Array`")

correct_vis.__doc__ = CORRECT_VIS_DOCS.substitute(
                        array_type=":class:`dask.array.Array`")

residual_vis.__doc__ = RESIDUAL_VIS_DOCS.substitute(
                        array_type=":class:`dask.array.Array`")
