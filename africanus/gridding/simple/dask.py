# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ...util.rtd import on_rtd
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
    from dask.array.core import getter
    import toolz

    from .gridding import (grid as nb_grid_fn, degrid as nb_degrid_fn)

    def grid(vis, uvw, flags, weights, ref_wave,
             convolution_filter, nx=1024, ny=1024):
        """ Documentation below """

        # Unfortunately necessary to introduce an extra dim
        # for atop to work properly
        def _grid_fn(*args, **kwargs):
            return nb_grid_fn(*args, **kwargs)[None, :, :, :]

        return da.core.atop(_grid_fn, ("row", "corr", "ny", "nx"),
                            vis, ("row", "chan", "corr"),
                            uvw, ("row", "(u,v,w)"),
                            flags, ("row", "chan", "corr"),
                            weights, ("row", "chan", "corr"),
                            ref_wave, ("chan",),
                            new_axes={"ny": ny, "nx": nx},
                            adjust_chunks={"row": lambda n: 1},
                            concatenate=True,
                            convolution_filter=convolution_filter,
                            dtype=np.complex64).sum(axis=0)

    def degrid(grid, uvw, weights, ref_wave, convolution_filter):
        """ Documentation below """
        return da.core.atop(nb_degrid_fn, ("row", "chan", "corr"),
                            grid, ("corr", "ny", "nx"),
                            uvw, ("row", "(u,v,w)"),
                            weights, ("row", "chan", "corr"),
                            ref_wave, ("chan",),
                            concatenate=True,
                            convolution_filter=convolution_filter,
                            dtype=np.complex64)

grid.__doc__ = """
dask wrapper for :func:`~africanus.gridding.simple.grid`.

Each ``row`` chunk of ``vis`` is gridded separately and the
resulting images are summed in a parallel reduction.

Parameters
----------
vis : :class:`dask.array.Array`
    complex64 visibility array of shape (row, chan, corr)
uvw : np.ndarray
    float64 array of UVW coordinates of shape (row, 3)
flags : :class:`dask.array.Array`
    flagged data array of shape (row, chan, corr).
    Any positive quantity will indicate that the corresponding
    visibility should be flagged.
    Set this to ``dask.array.zeros_like(vis, dtype=np.bool)``
    as default.
weights : :class:`dask.array.Array`
    float32 or float64 array of weights. Set this to
    ``dask.array.ones_like(vis, dtype=np.float32)`` as default.
ref_wave : :class:`dask.array.Array`
    float64 array of wavelengths of shape (chan,)
convolution_filter : :class:`~africanus.filters.ConvolutionFilter`
    Convolution filter
nx : integer, optional
    Size of the grid's X dimension
ny : integer, optional
    Size of the grid's Y dimension

Returns
-------
:class:`dask.array.Array`
    (corr, ny, nx) complex64 array of gridded visibilities
    constructed by summing convolutional weights.
"""

degrid.__doc__ = """
dask wrapper for :func:`~africanus.gridding.simple.degrid`.

Parameters
----------
grid : :class:`dask.array.Array`
    float or complex grid of visibilities
    of shape (corr, ny, nx)
uvw : :class:`dask.array.Array`
    float64 array of UVW coordinates of shape (row, 3)
weights : :class:`dask.array.Array`
    float32 or float64 array of weights. Set this to
    ``dask.array.ones_like(vis, dtype=np.float32)`` as default.
ref_wave : :class:`dask.array.Array`
    float64 array of wavelengths of shape (chan,)
convolution_filter :  :class:`~africanus.filters.ConvolutionFilter`
    Convolution Filter

Returns
-------
:class:`dask.array.Array`
    (chan, corr) complex64 visibilities
"""
