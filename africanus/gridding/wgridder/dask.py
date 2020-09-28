# -*- coding: utf-8 -*-

try:
    import dask.array as da
except ImportError as e:
    dask_import_error = e
else:
    dask_import_error = None

import numpy as np
from africanus.gridding.wgridder.vis2im import DIRTY_DOCS
from africanus.gridding.wgridder.im2vis import MODEL_DOCS
from africanus.gridding.wgridder.im2residim import RESIDUAL_DOCS
from africanus.gridding.wgridder.im2vis import _model_internal as model_np
from africanus.gridding.wgridder.vis2im import _dirty_internal as dirty_np
from africanus.gridding.wgridder.im2residim import (_residual_internal
                                                    as residual_np)
from africanus.util.requirements import requires_optional


def _model_wrapper(uvw, freq, model, freq_bin_idx, freq_bin_counts, cell,
                   weights, flag, celly, epsilon, nthreads, do_wstacking):

    return model_np(uvw[0], freq, model[0][0], freq_bin_idx, freq_bin_counts,
                    cell, weights, flag, celly, epsilon, nthreads,
                    do_wstacking)


@requires_optional('dask.array', dask_import_error)
def model(uvw, freq, image, freq_bin_idx, freq_bin_counts, cell,
          weights=None, flag=None, celly=None, epsilon=None, nthreads=1,
          do_wstacking=True):
    # determine output type
    complex_type = da.result_type(image, np.complex64)

    # set precision
    if epsilon is None:
        if image.dtype == np.float64:
            epsilon = 1e-7
        elif image.dtype == np.float32:
            epsilon = 1e-5
        else:
            raise ValueError("image of incorrect type")

    if celly is None:
        celly = cell

    if not nthreads:
        import multiprocessing
        nthreads = multiprocessing.cpu_count()

    if weights is None:
        weight_out = None
    else:
        weight_out = ('row', 'chan')

    if flag is None:
        flag_out = None
    else:
        flag_out = ('row', 'chan')

    vis = da.blockwise(_model_wrapper, ('row', 'chan'),
                       uvw, ('row', 'three'),
                       freq, ('chan',),
                       image, ('chan', 'nx', 'ny'),
                       freq_bin_idx, ('chan',),
                       freq_bin_counts, ('chan',),
                       cell, None,
                       weights, weight_out,
                       flag, flag_out,
                       celly, None,
                       epsilon, None,
                       nthreads, None,
                       do_wstacking, None,
                       adjust_chunks={'chan': freq.chunks[0]},
                       dtype=complex_type,
                       align_arrays=False)
    return vis


def _dirty_wrapper(uvw, freq, vis, freq_bin_idx, freq_bin_counts, nx, ny,
                   cell, weights, flag, celly, epsilon, nthreads,
                   do_wstacking):

    return dirty_np(uvw[0], freq, vis, freq_bin_idx, freq_bin_counts,
                    nx, ny, cell, weights, flag, celly, epsilon,
                    nthreads, do_wstacking)


@requires_optional('dask.array', dask_import_error)
def dirty(uvw, freq, vis, freq_bin_idx, freq_bin_counts, nx, ny, cell,
          weights=None, flag=None, celly=None, epsilon=None, nthreads=1,
          do_wstacking=True):

    # set precision
    if epsilon is None:
        if vis.dtype == np.complex128:
            epsilon = 1e-7
        elif vis.dtype == np.complex64:
            epsilon = 1e-5
        else:
            raise ValueError("vis of incorrect type")

    if celly is None:
        celly = cell

    if not nthreads:
        import multiprocessing
        nthreads = multiprocessing.cpu_count()

    if weights is None:
        weight_out = None
    else:
        weight_out = ('row', 'chan')

    if flag is None:
        flag_out = None
    else:
        flag_out = ('row', 'chan')

    img = da.blockwise(_dirty_wrapper, ('row', 'chan', 'nx', 'ny'),
                       uvw, ('row', 'three'),
                       freq, ('chan',),
                       vis, ('row', 'chan'),
                       freq_bin_idx, ('chan',),
                       freq_bin_counts, ('chan',),
                       nx, None,
                       ny, None,
                       cell, None,
                       weights, weight_out,
                       flag, flag_out,
                       celly, None,
                       epsilon, None,
                       nthreads, None,
                       do_wstacking, None,
                       adjust_chunks={'chan': freq_bin_idx.chunks[0],
                                      'row': (1,)*len(vis.chunks[0])},
                       new_axes={"nx": nx, "ny": ny},
                       dtype=weights.dtype,
                       align_arrays=False)

    return img.sum(axis=0)


def _residual_wrapper(uvw, freq, model, vis, freq_bin_idx, freq_bin_counts,
                      cell, weights, flag, celly, epsilon, nthreads,
                      do_wstacking):

    return residual_np(uvw[0], freq, model, vis, freq_bin_idx,
                       freq_bin_counts, cell, weights, flag, celly, epsilon,
                       nthreads, do_wstacking)


@requires_optional('dask.array', dask_import_error)
def residual(uvw, freq, image, vis, freq_bin_idx, freq_bin_counts, cell,
             weights=None, flag=None, celly=None, epsilon=None,
             nthreads=1, do_wstacking=True):

    # set precision
    if epsilon is None:
        if image.dtype == np.float64:
            epsilon = 1e-7
        elif image.dtype == np.float32:
            epsilon = 1e-5
        else:
            raise ValueError("image of incorrect type")

    if celly is None:
        celly = cell

    if not nthreads:
        import multiprocessing
        nthreads = multiprocessing.cpu_count()

    if weights is None:
        weight_out = None
    else:
        weight_out = ('row', 'chan')

    if flag is None:
        flag_out = None
    else:
        flag_out = ('row', 'chan')

    img = da.blockwise(_residual_wrapper, ('row', 'chan', 'nx', 'ny'),
                       uvw, ('row', 'three'),
                       freq, ('chan',),
                       image, ('chan', 'nx', 'ny'),
                       vis, ('row', 'chan'),
                       freq_bin_idx, ('chan',),
                       freq_bin_counts, ('chan',),
                       cell, None,
                       weights, weight_out,
                       flag, flag_out,
                       celly, None,
                       epsilon, None,
                       nthreads, None,
                       do_wstacking, None,
                       adjust_chunks={'chan': freq_bin_idx.chunks[0],
                                      'row': (1,)*len(vis.chunks[0])},
                       dtype=image.dtype,
                       align_arrays=False)
    return img.sum(axis=0)


model.__doc__ = MODEL_DOCS.substitute(
                    array_type=":class:`dask.array.Array`")
dirty.__doc__ = DIRTY_DOCS.substitute(
                    array_type=":class:`dask.array.Array`")
residual.__doc__ = RESIDUAL_DOCS.substitute(
                     array_type=":class:`dask.array.Array`")
