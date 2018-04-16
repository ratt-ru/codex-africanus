# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from operator import mul

import numba
import numpy as np

from ...util.docs import on_rtd


@numba.jit(nopython=True, nogil=True, cache=True)
def _nb_grid(vis, uvw, flags, weights, ref_wave,
             convolution_filter, grid):
    """
    See :func:"~africanus.gridding.simple.gridding._grid" for
    documentation.
    """
    cf = convolution_filter

    nrow, nchan = vis.shape[0:2]
    assert nchan == ref_wave.shape[0]
    corrs = vis.shape[2:]

    ny, nx = grid.shape[0:2]
    flat_corrs = grid.shape[2]

    # Flatten correlation dimension for easier loop handling
    fvis = vis.reshape((nrow,nchan,flat_corrs))
    fflags = flags.reshape((nrow,nchan,flat_corrs))
    fweights = weights.reshape((nrow,nchan,flat_corrs))

    filter_index = np.arange(-cf.half_sup, cf.half_sup+1)

    for r in range(uvw.shape[0]):                 # row (vis)
        for f in range(vis.shape[1]):             # channel (freq)
            scaled_u = uvw[r, 0] / ref_wave[f]
            scaled_v = uvw[r, 1] / ref_wave[f]

            disc_u = int(np.round(scaled_u))
            disc_v = int(np.round(scaled_v))

            extent_v = disc_v + ny // 2
            extent_u = disc_u + nx // 2

            # Out of bounds check
            if (extent_v + cf.half_sup >= ny or
                extent_u + cf.half_sup >= nx or
                extent_v - cf.half_sup < 0 or
                    extent_u - cf.half_sup < 0):
                continue

            # One plus half support
            one_half_sup = 1 + cf.half_sup

            # Compute fractional u and v
            base_frac_u = one_half_sup + disc_u - scaled_u
            base_frac_v = one_half_sup + disc_v - scaled_v

            frac_u = int(base_frac_u*cf.oversample)
            frac_v = int(base_frac_v*cf.oversample)

            # Iterate over v/y
            for conv_v in filter_index:
                v_tap = cf.filter_taps[conv_v*cf.oversample + frac_v]
                grid_v = disc_v + conv_v + ny // 2

                # Iterate over u/x
                for conv_u in filter_index:
                    u_tap = cf.filter_taps[conv_u*cf.oversample + frac_u]
                    conv_weight = v_tap*u_tap
                    grid_u = disc_u + conv_u + nx // 2

                    for c in range(flat_corrs):      # correlation
                        # Ignore flagged correlations
                        if fflags[r, f, c] > 0:
                            continue

                        # Grid the visibility
                        grid[grid_v, grid_u, c] += (fvis[r, f, c] *
                                                    conv_weight *
                                                    fweights[r, f, c])

    return grid.reshape((ny, nx) + corrs)


def grid(vis, uvw, flags, weights, ref_wave,
         convolution_filter,
         nx=1024, ny=1024,
         grid=None):
    """
    Convolutional gridder which grids visibilities ``vis``
    at the specified ``uvw`` coordinates and
    ``ref_wave`` reference wavelengths using
    the specified ``convolution_filter``.

    Parameters
    ----------
    vis : np.ndarray
        complex visibility array of shape :code:`(row, chan, corr_1, corr_2)`
    uvw : np.ndarray
        float64 array of UVW coordinates of shape :code:`(row, 3)`
    weights : np.ndarray
        float32 or float64 array of weights. Set this to
        ``np.ones_like(vis, dtype=np.float32)`` as default.
    flags : np.ndarray
        flagged array of shape :code:`(row, chan, corr_1, corr_2)`.
        Any positive quantity will indicate that the corresponding
        visibility should be flagged.
        Set to ``np.zero_like(vis, dtype=np.bool)`` as default.
    ref_wave : np.ndarray
        float64 array of wavelengths of shape :code:`(chan,)`
    convolution_filter :  :class:`~africanus.filters.ConvolutionFilter`
        Convolution filter
    nx : integer, optional
        Size of the grid's X dimension
    ny : integer, optional
        Size of the grid's Y dimension
    grid : np.ndarray, optional
        complex64/complex128 array of shape :code:`(ny, nx, corr_1, corr_2)`
        If supplied, this array will be used as the gridding target,
        and ``nx`` and ``ny`` will be derived from this grid's
        dimensions.

    Returns
    -------
    np.ndarray
        :code:`(ny, nx, corr_1, corr_2)` complex ndarray of gridded visibilities
    """

    # Flatten the correlation dimensions
    corrs = vis.shape[2:]
    flat_corrs = (reduce(mul, corrs),)

    # Create grid of flatten correlations or reshape
    if grid is None:
        grid = np.zeros((ny, nx) + flat_corrs, dtype=vis.dtype)
    else:
        ny, nx = grid.shape[0:2]
        grid = grid.reshape((ny, nx) + flat_corrs)

    return _nb_grid(vis, uvw, flags, weights, ref_wave,
                    convolution_filter, grid)

def _degrid(grid, uvw, weights, ref_wave, convolution_filter):
    """
    Convolutional degridder (continuum)

    Parameters
    ----------
    grid : np.ndarray
        float or complex grid of visibilities
        of shape :code:`(ny, nx, corr_1, corr_2)`
    uvw : np.ndarray
        float64 array of UVW coordinates of shape :code:`(row, 3)`
    weights : np.ndarray
        float32 or float64 array of weights. Set this to
        ``np.ones_like(vis, dtype=np.float32)`` as default.
    ref_wave : np.ndarray
        float64 array of wavelengths of shape :code:`(chan,)`
    convolution_filter :  :class:`~africanus.filters.ConvolutionFilter`
        Convolution Filter

    Returns
    -------
    np.ndarray
        :code:`(row, chan, corr_1, corr_2)` complex ndarray of visibilities
    """
    cf = convolution_filter

    ny, nx = grid.shape[0:2]
    corrs = grid.shape[2:]
    nchan = ref_wave.shape[0]
    nrow = uvw.shape[0]

    flat_corrs = 1

    for corr in corrs:
        flat_corrs *= corr


    vis = np.zeros((nrow, nchan, flat_corrs), dtype=np.complex64)
    grid = grid.reshape((ny, nx, flat_corrs))
    weights = weights.reshape((nrow, nchan, flat_corrs))

    assert vis.shape[1] == ref_wave.shape[0]
    filter_index = np.arange(-cf.half_sup, cf.half_sup+1)

    for r in range(uvw.shape[0]):                 # row (vis)
        for f in range(vis.shape[1]):             # channel

            scaled_u = uvw[r, 0] / ref_wave[f]
            scaled_v = uvw[r, 1] / ref_wave[f]

            disc_u = int(round(scaled_u))
            disc_v = int(round(scaled_v))

            extent_v = disc_v + ny // 2
            extent_u = disc_u + nx // 2

            # Out of bounds check
            if (extent_v + cf.half_sup >= ny or
                extent_u + cf.half_sup >= nx or
                extent_v - cf.half_sup < 0 or
                    extent_u - cf.half_sup < 0):
                continue

            # One plus half support
            one_half_sup = 1 + cf.half_sup

            # Compute fractional u and v
            base_frac_u = one_half_sup + disc_u - scaled_u
            base_frac_v = one_half_sup + disc_v - scaled_v

            frac_u = int(base_frac_u*cf.oversample)
            frac_v = int(base_frac_v*cf.oversample)

            for conv_v in filter_index:
                v_tap = cf.filter_taps[conv_v*cf.oversample + frac_v]
                grid_v = disc_v + conv_v + ny // 2

                for conv_u in filter_index:
                    u_tap = cf.filter_taps[conv_u*cf.oversample + frac_u]
                    conv_weight = v_tap * u_tap
                    grid_u = disc_u + conv_u + nx // 2

                    # Correlation
                    for c in range(flat_corrs):
                        vis[r, f, c] += (grid[grid_v, grid_u, c] *
                                         conv_weight *
                                         weights[r, f, c])

    return vis.reshape((nrow, nchan) + corrs)

# jit the functions if this is not RTD otherwise
# use the private funcs for generating docstrings


if not on_rtd():
    degrid = numba.jit(nopython=True, nogil=True, cache=True)(_degrid)
    # degrid = _degrid
else:
    degrid = _degrid
