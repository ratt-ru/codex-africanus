# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import wraps

from africanus.calibration.phase_only.phase_only import COMPUTE_JHJ_DOCS
from africanus.calibration.phase_only.phase_only import COMPUTE_JHR_DOCS
from africanus.calibration.utils import check_type
from africanus.calibration.phase_only.phase_only import (compute_jhj
                                                         as np_compute_jhj)
from africanus.calibration.phase_only.phase_only import (compute_jhr
                                                         as np_compute_jhr)
from africanus.util.requirements import requires_optional

try:
    from dask.array.core import blockwise
except ImportError as e:
    dask_import_error = e
else:
    dask_import_error = None

DIAG_DIAG = 0
DIAG = 1
FULL = 2


@wraps(np_compute_jhj)
def _compute_jhj_wrapper(time_bin_indices, time_bin_counts, antenna1,
                         antenna2, jones, model, flag):
    return np_compute_jhj(time_bin_indices, time_bin_counts, antenna1,
                          antenna2, jones, model, flag)


@requires_optional('dask.array', dask_import_error)
def compute_jhj(time_bin_indices, time_bin_counts, antenna1,
                antenna2, jones, model, flag):

    mode = check_type(jones, model, vis_type='model')

    if mode != DIAG_DIAG:
        raise NotImplementedError("Only DIAG-DIAG case has been implemented")

    jones_shape = ('row', 'ant', 'chan', 'dir', 'corr')
    vis_shape = ('row', 'chan', 'corr')
    model_shape = ('row', 'chan', 'dir', 'corr')
    return blockwise(_compute_jhj_wrapper, jones_shape,
                     time_bin_indices, ('row',),
                     time_bin_counts, ('row',),
                     antenna1, ('row',),
                     antenna2, ('row',),
                     jones, jones_shape,
                     model, model_shape,
                     flag, vis_shape,
                     adjust_chunks={"row": antenna1.chunks[0]},
                     new_axes={"corr2": 2},  # why?
                     dtype=model.dtype,
                     align_arrays=False)


@wraps(np_compute_jhr)
def _compute_jhr_wrapper(time_bin_indices, time_bin_counts, antenna1,
                         antenna2, jones, residual, model, flag):
    return np_compute_jhr(time_bin_indices, time_bin_counts, antenna1,
                          antenna2, jones, residual, model, flag)


@requires_optional('dask.array', dask_import_error)
def compute_jhr(time_bin_indices, time_bin_counts, antenna1,
                antenna2, jones, residual, model, flag):

    mode = check_type(jones, residual)

    if mode != DIAG_DIAG:
        raise NotImplementedError("Only DIAG-DIAG case has been implemented")

    jones_shape = ('row', 'ant', 'chan', 'dir', 'corr')
    vis_shape = ('row', 'chan', 'corr')
    model_shape = ('row', 'chan', 'dir', 'corr')
    return blockwise(_compute_jhr_wrapper, jones_shape,
                     time_bin_indices, ('row',),
                     time_bin_counts, ('row',),
                     antenna1, ('row',),
                     antenna2, ('row',),
                     jones, jones_shape,
                     residual, vis_shape,
                     model, model_shape,
                     flag, vis_shape,
                     adjust_chunks={"row": antenna1.chunks[0]},
                     new_axes={"corr2": 2},  # why?
                     dtype=model.dtype,
                     align_arrays=False)


compute_jhj.__doc__ = COMPUTE_JHJ_DOCS.substitute(
                        array_type=":class:`dask.array.Array`")

compute_jhr.__doc__ = COMPUTE_JHR_DOCS.substitute(
                        array_type=":class:`dask.array.Array`")
