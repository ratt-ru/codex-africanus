# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from operator import mul

try:
    # in python3, reduce is in functools
    from functools import reduce
except ImportError:
    # I guess this is python2
    pass

import numba
import numpy as np

_ARCSEC2RAD = np.deg2rad(1.0/(60*60))


@numba.jit(nopython=True, nogil=True, cache=True)
def _all_flagged(fflags):
    # Fail if anything is not flagged
    for flag in fflags:
        if flag == 0:
            return False

    return True


@numba.jit(nopython=True, nogil=True, cache=True)
def numba_grid(vis, uvw, flags, weights, ref_wave,
               convolution_filter, cell_size, grid):
    """
    See :func:"~africanus.gridding.simple.gridding.grid" for
    documentation.
    """
    cf = convolution_filter

    # Shape checks
    assert vis.shape[0] == uvw.shape[0] == flags.shape[0] == weights.shape[0]
    assert vis.shape[1] == flags.shape[1] == weights.shape[1]
    assert vis.shape[2] == flags.shape[2] == weights.shape[2]

    nrow, nchan = vis.shape[0:2]
    assert nchan == ref_wave.shape[0]
    corrs = vis.shape[2:]

    ny, nx = grid.shape[0:2]
    flat_corrs = grid.shape[2]

    # Similarity Theorem
    # https://www.cv.nrao.edu/course/astr534/FTSimilarity.html
    # Scale UV coordinates
    # Note u => x and v => y
    u_scale = _ARCSEC2RAD * cell_size * nx
    v_scale = _ARCSEC2RAD * cell_size * ny

    # Flatten correlation dimension for easier loop handling
    fvis = vis.reshape((nrow, nchan, flat_corrs))
    fflags = flags.reshape((nrow, nchan, flat_corrs))
    fweights = weights.reshape((nrow, nchan, flat_corrs))

    oversample = cf.oversampling
    base_os = oversample // 2 if oversample % 2 == 1 else (oversample // 2) - 1
    half_support = cf.full_support // 2
    half_x = nx // 2
    half_y = ny // 2

    for r in range(uvw.shape[0]):                 # row (vis)
        scaled_u = uvw[r, 0] * u_scale
        scaled_v = uvw[r, 1] * v_scale

        for f in range(vis.shape[1]):             # channel (freq)
            sub_flags = fflags[r, f]

            # Continue early if all correlations are flagged
            if _all_flagged(sub_flags):
                continue

            # Exact floating point UV coordinates in the grid
            exact_u = half_x + (scaled_u / ref_wave[f])
            exact_v = half_y + (scaled_v / ref_wave[f])

            # Discretised UV coordinates, snapped to nearest grid point
            disc_u = int(np.round(exact_u))
            disc_v = int(np.round(exact_v))

            # Filter extents in U and V
            lower_u = disc_u - half_support
            lower_v = disc_v - half_support
            upper_u = disc_u + half_support + 1  # Exclusive
            upper_v = disc_v + half_support + 1  # Exclusive

            # Next visibility if outside extents
            if (lower_u < 0 or lower_v < 0 or upper_u > nx or upper_v > ny):
                continue

            # Compute fractional u and v
            frac_u = exact_u - disc_u
            frac_v = exact_v - disc_v

            # Base oversampling index in the filter
            # [-0.5, 0.5]*oversample
            base_os_u = int(np.round(frac_u*oversample))
            base_os_v = int(np.round(frac_u*oversample))

            # Normalise the index from
            # [-2, -1,  0,  1,  2, -2, -1,  0   1,  2, -2, -1,  0,  1,  2] to
            # [ 0,  1,  2,  3,  4,  0,  1,  2,  3,  4,  0,  1,  2,  3,  4]
            base_os_u = (base_os_u + ((3 * oversample) // 2)) % oversample
            base_os_v = (base_os_v + ((3 * oversample) // 2)) % oversample

            sub_vis = vis[r, f]
            sub_weights = weights[r, f]

            # Iterate over v/y
            for vi, grid_v in enumerate(range(lower_v, upper_v)):
                v_filter = cf.filter[base_os_v + vi*oversample, :]
                sub_grid_v = grid[grid_v, :, :]

                # Iterate over u/x
                for ui, grid_u in enumerate(range(lower_u, upper_u)):
                    conv_weight = v_filter[base_os_u + ui*oversample]
                    sub_grid_u = sub_grid_v[grid_u, :]

                    for c in range(flat_corrs):      # correlation
                        # Ignore flagged correlations
                        if sub_flags[c] > 0:
                            continue

                        # Grid the visibility
                        sub_grid_u[c] += sub_vis[c]*conv_weight*sub_weights[c]

    return grid.reshape((ny, nx) + corrs)


def grid(vis, uvw, flags, weights, ref_wave,
         convolution_filter,
         cell_size,
         nx=1024, ny=1024,
         grid=None):
    """
    Convolutional gridder which grids visibilities ``vis``
    at the specified ``uvw`` coordinates and
    ``ref_wave`` reference wavelengths using
    the specified ``convolution_filter``.

    Variable numbers of correlations are supported.

    * :code:`(row, chan, corr_1, corr_2)` ``vis`` will result in a
      :code:`(ny, nx, corr_1, corr_2)` ``grid``.
    * :code:`(row, chan, corr_1)` ``vis`` will result in a
      :code:`(ny, nx, corr_1)` ``grid``.

    Parameters
    ----------
    vis : np.ndarray
        complex visibility array of shape :code:`(row, chan, corr_1, corr_2)`
    uvw : np.ndarray
        float64 array of UVW coordinates of shape :code:`(row, 3)`
        in wavelengths.
    weights : np.ndarray
        float32 or float64 array of weights. Set this to
        ``np.ones_like(vis, dtype=np.float32)`` as default.
    flags : np.ndarray
        flagged array of shape :code:`(row, chan, corr_1, corr_2)`.
        Any positive quantity will indicate that the corresponding
        visibility should be flagged.
        Set to ``np.zeros_like(vis, dtype=np.bool)`` as default.
    ref_wave : np.ndarray
        float64 array of wavelengths of shape :code:`(chan,)`
    convolution_filter :  :class:`~africanus.filters.ConvolutionFilter`
        Convolution filter
    cell_size : float
        Cell size in arcseconds.
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
        :code:`(ny, nx, corr_1, corr_2)` complex ndarray of
        gridded visibilities. The number of correlations may vary,
        depending on the shape of vis.
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

    return numba_grid(vis, uvw, flags, weights, ref_wave,
                      convolution_filter, cell_size, grid)


@numba.jit(nopython=True, nogil=True, cache=True)
def numba_degrid(grid, uvw, weights, ref_wave,
                 convolution_filter, cell_size, vis):
    """
    See :func:"~africanus.gridding.simple.gridding.degrid" for
    documentation.
    """
    cf = convolution_filter
    ny, nx, flat_corrs = grid.shape

    # Similarity Theorem
    # https://www.cv.nrao.edu/course/astr534/FTSimilarity.html
    # Scale UV coordinates
    # Note u => x and v => y
    u_scale = _ARCSEC2RAD * cell_size * nx
    v_scale = _ARCSEC2RAD * cell_size * ny
    oversample = cf.oversampling
    base_os = oversample // 2 if oversample % 2 == 1 else (oversample // 2) - 1
    half_support = cf.full_support // 2
    half_x = nx // 2
    half_y = ny // 2

    for r in range(uvw.shape[0]):                 # row (vis)
        scaled_u = uvw[r, 0] * u_scale
        scaled_v = uvw[r, 1] * v_scale

        for f in range(vis.shape[1]):             # channel (freq)
            # Exact floating point UV coordinates in the grid
            exact_u = half_x + (scaled_u / ref_wave[f])
            exact_v = half_y + (scaled_v / ref_wave[f])

            # Discretised UV coordinates, snapped to nearest grid point
            disc_u = int(np.round(exact_u))
            disc_v = int(np.round(exact_v))

            # Filter extents in U and V
            lower_u = disc_u - half_support
            lower_v = disc_v - half_support
            upper_u = disc_u + half_support + 1  # Exclusive
            upper_v = disc_v + half_support + 1  # Exclusive

            # Next visibility if outside extents
            if (lower_u < 0 or lower_v < 0 or upper_u > nx or upper_v > ny):
                continue

            # Compute fractional u and v
            frac_u = exact_u - disc_u
            frac_v = exact_v - disc_v

            # Base oversampling index in the filter
            # [-0.5, 0.5]*oversample
            base_os_u = int(np.round(frac_u*oversample))
            base_os_v = int(np.round(frac_u*oversample))

            # Normalise the index from
            # [-2, -1,  0,  1,  2, -2, -1,  0   1,  2, -2, -1,  0,  1,  2] to
            # [ 0,  1,  2,  3,  4,  0,  1,  2,  3,  4,  0,  1,  2,  3,  4]
            base_os_u = (base_os_u + ((3 * oversample) // 2)) % oversample
            base_os_v = (base_os_v + ((3 * oversample) // 2)) % oversample

            sub_vis = vis[r, f]
            sub_weights = weights[r, f]

            # Iterate over v/y
            for vi, grid_v in enumerate(range(lower_v, upper_v)):
                v_filter = cf.filter[base_os_v + vi*oversample, :]
                sub_grid_v = grid[grid_v, :, :]

                # Iterate over u/x
                for ui, grid_u in enumerate(range(lower_u, upper_u)):
                    conv_weight = v_filter[base_os_u + ui*oversample]
                    sub_grid_u = sub_grid_v[grid_u, :]

                    for c in range(flat_corrs):
                        sub_vis[c] += sub_grid_u[c]*conv_weight*sub_weights[c]

    return vis


def degrid(grid, uvw, weights, ref_wave,
           convolution_filter, cell_size, dtype=np.complex64):
    """
    Convolutional degridder (continuum)

    Variable numbers of correlations are supported.

    * :code:`(ny, nx, corr_1, corr_2)` ``grid`` will result in a
      :code:`(row, chan, corr_1, corr_2)` ``vis``

    * :code:`(ny, nx, corr_1)` ``grid`` will result in a
      :code:`(row, chan, corr_1)` ``vis``

    Parameters
    ----------
    grid : np.ndarray
        float or complex grid of visibilities
        of shape :code:`(ny, nx, corr_1, corr_2)`
    uvw : np.ndarray
        float64 array of UVW coordinates of shape :code:`(row, 3)`
        in wavelengths.
    weights : np.ndarray
        float32 or float64 array of weights. Set this to
        ``np.ones_like(vis, dtype=np.float32)`` as default.
    ref_wave : np.ndarray
        float64 array of wavelengths of shape :code:`(chan,)`
    convolution_filter :  :class:`~africanus.filters.ConvolutionFilter`
        Convolution Filter
    cell_size : float
        Cell size in arcseconds.
    dtype : :class:`numpy.dtype`
        Data type of the visibilities

    Returns
    -------
    np.ndarray
        :code:`(row, chan, corr_1, corr_2)` complex ndarray of visibilities
    """
    nrow = uvw.shape[0]
    nchan = ref_wave.shape[0]
    corrs = flat_corrs = grid.shape[2:]

    # Flatten if necessary
    if len(corrs) > 1:
        flat_corrs = (reduce(mul, corrs),)
        grid = grid.reshape(grid.shape[:2] + flat_corrs)
        weights = weights.reshape(weights.shape[:2] + flat_corrs)

    vis = np.zeros((nrow, nchan) + flat_corrs, dtype=dtype)

    vis = numba_degrid(grid, uvw, weights, ref_wave,
                       convolution_filter, cell_size, vis)

    return vis.reshape(weights.shape[:2] + corrs)
