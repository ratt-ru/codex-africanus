# -*- coding: utf-8 -*-


from functools import reduce
from operator import mul

import numpy as np

from africanus.gridding.simple.gridding import (grid as np_grid_fn,
                                                degrid as np_degrid_fn)
from africanus.util.docs import mod_docs
from africanus.util.requirements import requires_optional

try:
    import dask.array as da
except ImportError as e:
    da_import_error = e
else:
    da_import_error = None


# Unfortunately necessary to introduce an extra dim
# for blockwise to work properly
def _grid_fn(vis, uvw, flags, weights, ref_wave, convolution_filter,
             cell_size, nx, ny):
    return np_grid_fn(vis[0], uvw[0], flags[0], weights[0],
                      ref_wave[0], convolution_filter,
                      cell_size,
                      nx=nx, ny=ny)[None, :]


@requires_optional('dask.array', da_import_error)
def grid(vis, uvw, flags, weights, ref_wave,
         convolution_filter, cell_size, nx=1024, ny=1024):
    """ Documentation below """

    # Creation correlation dimension strings for each correlation
    corrs = tuple('corr-%d' % i for i in range(len(vis.shape[2:])))

    # Get grids, stacked by row
    grids = da.core.blockwise(_grid_fn, ("row", "ny", "nx") + corrs,
                              vis, ("row", "chan") + corrs,
                              uvw, ("row", "(u,v,w)"),
                              flags, ("row", "chan") + corrs,
                              weights, ("row", "chan") + corrs,
                              ref_wave, ("chan",),
                              new_axes={"ny": ny, "nx": nx},
                              adjust_chunks={"row": 1},
                              convolution_filter=convolution_filter,
                              cell_size=cell_size, ny=ny, nx=nx,
                              dtype=vis.dtype)

    # Sum grids over the row dimension to produce (ny, nx, corr_1, corr_2)
    return grids.sum(axis=0)


@requires_optional('dask.array', da_import_error)
def degrid(grid, uvw, weights, ref_wave, convolution_filter, cell_size):
    """ Documentation below """

    grid_flat_corrs = reduce(mul, grid.shape[2:])
    weight_flat_corrs = reduce(mul, weights.shape[2:])

    assert grid_flat_corrs == weight_flat_corrs
    assert uvw.shape[0] == weights.shape[0]
    assert weights.shape[1] == ref_wave.shape[0]

    # Creation correlation dimension strings for each correlation
    corrs = tuple('corr-%d' % i for i in range(len(grid.shape[2:])))

    return da.core.blockwise(np_degrid_fn, ("row", "chan") + corrs,
                             grid, ("ny", "nx") + corrs,
                             uvw, ("row", "(u,v,w)"),
                             weights, ("row", "chan") + corrs,
                             ref_wave, ("chan",),
                             concatenate=True,
                             convolution_filter=convolution_filter,
                             cell_size=cell_size,
                             dtype=np.complex64)


grid.__doc__ = mod_docs(np_grid_fn.__doc__,
                        [(":class:`numpy.ndarray`",
                            ":class:`dask.array.Array`"),
                         ("np.ones_like", "da.ones_like"),
                         ("np.zeros_like", "da.zeros_like")])

degrid.__doc__ = mod_docs(np_degrid_fn.__doc__,
                          [(":class:`numpy.ndarray`",
                            ":class:`dask.array.Array`"),
                           ("np.ones_like", "da.ones_like"),
                           ("np.zeros_like", "da.zeros_like")])
