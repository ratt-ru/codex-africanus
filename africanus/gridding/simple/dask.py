# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .gridding import (grid as np_grid_fn, degrid as np_degrid_fn)
from ...util.docs import on_rtd, mod_docs
from ...util.requirements import have_packages, MissingPackageException

_package_requirements = ('dask.array', 'toolz')
have_requirements = have_packages(*_package_requirements)

if not have_requirements or on_rtd():
    def grid(vis, uvw, flags, weights, ref_wave,
             convolution_filter, nx=1024, ny=1024):
        raise MissingPackageException(*_package_requirements)

    def degrid(grid, uvw, weights, ref_wave, convolution_filter):
        raise MissingPackageException(*_package_requirements)
else:
    import numpy as np
    import dask.array as da

    def grid(vis, uvw, flags, weights, ref_wave,
             convolution_filter, nx=1024, ny=1024):
        """ Documentation below """

        # Creation correlation dimension strings for each correlation
        corrs = tuple('corr-%d' for i in range(len(vis.shape[2:])))

        # Unfortunately necessary to introduce an extra dim
        # for atop to work properly
        def _grid_fn(vis, uvw, flags, weights, ref_wave, convolution_filter):
            return np_grid_fn(vis[0], uvw[0], flags[0], weights[0],
                                ref_wave[0], convolution_filter,
                                nx=nx, ny=ny)[None,:]

        # Get grids, stacked by row
        grids = da.core.atop(_grid_fn, ("row", "ny", "nx") + corrs,
                            vis, ("row", "chan") + corrs,
                            uvw, ("row", "(u,v,w)"),
                            flags, ("row", "chan") + corrs,
                            weights, ("row", "chan") + corrs,
                            ref_wave, ("chan",),
                            new_axes={"ny": ny, "nx": nx},
                            adjust_chunks={"row": 1},
                            convolution_filter=convolution_filter,
                            dtype=np.complex64)

        # Sum grids over the row dimension to produce (ny, nx, corr_1, corr_2)
        return grids.sum(axis=0)

    def degrid(grid, uvw, weights, ref_wave, convolution_filter):
        """ Documentation below """

        # Creation correlation dimension strings for each correlation
        corrs = tuple('corr-%d' for i in range(len(grid.shape[2:])))

        return da.core.atop(np_degrid_fn, ("row", "chan") + corrs,
                            grid, ("ny", "nx") + corrs,
                            uvw, ("row", "(u,v,w)"),
                            weights, ("row", "chan") + corrs,
                            ref_wave, ("chan",),
                            concatenate=True,
                            convolution_filter=convolution_filter,
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



