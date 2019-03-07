from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from operator import mul

import numpy as np

from africanus.compatibility import reduce
from africanus.util.numba import jit
from africanus.gridding.simple.gridding import (
                numba_grid as simple_numba_grid,
                numba_degrid as simple_numba_degrid)


def w_stacking_layers(w_min, w_max, l, m):
    r"""
    Computes the number of w-layers given the minimum and
    maximum W coordinates, as well as the l and m coordinates.

    .. math::

        N_{wlay} >> 2 \pi \left(w_{max} - w_{min} \right)
        \underset{l, m}{\max}\left(1 - \sqrt{1 - l^2 - m^2}\right)

    Parameters
    ----------
    w_min : float
        Minimum W coordinate in wavelengths.
    w_max : float
        Maximum W coordinate in wavelengths.
    l : :class:`numpy.ndarray`
        l coordinates
    m : :class:`numpy.ndarray`
        m coordinates

    Returns
    -------
    int
        Number of w-layers
    """
    max_val = (1.0 - np.sqrt(1 - l[None, :]**2 - m[:, None]**2)).max()
    return np.ceil(2*np.pi*(w_max - w_min)*max_val).astype(np.int32).item()


def w_stacking_bins(w_min, w_max, w_layers):
    r"""
    Returns the W coordinate bins appropriate for the observation parameters,
    given the minimum and maximum W coordinates and the number of W layers.

    W coordinates can be binned by calling

    .. code-block:: python

        w_bins = np.digitize(w, bins) - 1

    Notes
    -----

    A small epsilon is added to ``w_max`` to force this
    W coordinate into the last bin.

    Parameters
    ----------
    w_min : float
        Minimum W coordinate in wavelengths.
    w_max : float
        Maximum W coordinate in wavelengths.
    w_layers : int
        Number of w layers

    Returns
    -------
    :class:`numpy.ndarray`
        W-coordinate bins of shape :code:`(nw + 1,)`.
    """
    return np.linspace(w_min, w_max + 1e-12, w_layers + 1)


@jit(nopython=True, nogil=True, cache=True)
def w_stacking_centroids(w_bins):
    return 0.5*(w_bins[:-1] + w_bins[1:])


WSTACK_DOCS = r"""
Returns the W coordinate centroids for each
W layer. Computed from bins produced by
:func:`w_stacking_bins`.

Parameters
----------
w_bins : :class:`numpy.ndarray`
    W stacking bins of shape :code:`(nw + 1,)`

Returns
-------
:class:`numpy.ndarray`
    W-coordinate centroids of shape :code:`(nw,)`
    in wavelengths.
"""


try:
    w_stacking_centroids.__doc__ = WSTACK_DOCS
except AttributeError:
    pass


@jit(nopython=True, nogil=True, cache=True)
def w_bin_masks(uvw, w_bins):
    indices = np.digitize(uvw[:, 2], w_bins) - 1
    return [i == indices for i in range(w_bins.shape[0])]


@jit(nopython=True, nogil=True, cache=True)
def numba_grid(vis, uvw, flags, weights, ref_wave,
               convolution_filter, w_bins, cell_size, grids):

    assert len(grids) == w_bins.shape[0] - 1
    bin_indices = np.digitize(uvw[:, 2], w_bins) - 1

    if np.any(bin_indices < 0):
        raise ValueError("bin_index < 0")

    if np.any(bin_indices >= len(grids)):
        raise ValueError("bin_index >= len(grids)")

    for i, grid in enumerate(grids):
        # The row mask for this layer
        mask = bin_indices == i

        # Nothing to grid
        if np.sum(mask) == 0:
            continue

        simple_numba_grid(vis[mask, ...],
                          uvw[mask, ...],
                          flags[mask, ...],
                          weights[mask, ...],
                          ref_wave,
                          convolution_filter,
                          cell_size,
                          grid)

    return grids


def grid(vis, uvw, flags, weights, ref_wave,
         convolution_filter, w_bins,
         cell_size,
         nx=1024, ny=1024,
         grids=None):
    """
    Convolutional W-stacking gridder.

    This function grids visibilities ``vis`` onto multiple
    grids, each associated with a W-layer defined by ``w_bins``.
    The W coordinate of the ``uvw`` array is used to bin the visibility
    into the appropriate grid.

    Variable numbers of correlations are supported.

    * :code:`(row, chan, corr_1, corr_2)` ``vis`` will result in a
      :code:`(ny, nx, corr_1, corr_2)` ``grid``.
    * :code:`(row, chan, corr_1)` ``vis`` will result in a
      :code:`(ny, nx, corr_1)` ``grid``.

    Parameters
    ----------
    vis : :class:`numpy.ndarray`
        complex visibility array of shape :code:`(row, chan, corr_1, corr_2)`
    uvw : :class:`numpy.ndarray`
        float64 array of UVW coordinates of shape :code:`(row, 3)`
    weights : :class:`numpy.ndarray`
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
    w_bins : :class:`numpy.ndarray`
        W coordinate bins of shape :code:`(nw + 1,)`
    cell_size : float
        Cell size in arcseconds.
    nx : integer, optional
        Size of the grid's X dimension
    ny : integer, optional
        Size of the grid's Y dimension
    grids : list of np.ndarray, optional
        list of complex arrays of length :code:`nw`,
        each with shape :code:`(ny, nx, corr_1, corr_2)`.
        If supplied, this array will be used as the gridding target,
        and ``nx`` and ``ny`` will be derived from the grid's
        dimensions.

    Returns
    -------
    list of np.ndarray
        list of complex arrays of gridded visibilities, of length :code:`nw`,
        each with shape :code:`(ny, nx, corr_1, corr_2)`.
        The number of correlations may vary,
        depending on the shape of vis.
    """
    corrs = vis.shape[2:]

    # Create grid of flatten correlations or reshape
    if grids is None:
        nw = w_bins.shape[0] - 1
        grids = [np.zeros((ny, nx) + corrs, dtype=vis.dtype)
                 for _ in range(nw)]
    elif not isinstance(grids, list):
        grids = [grids]

    # Flatten the correlation dimensions
    flat_corrs = (reduce(mul, corrs),)
    grids = [g.reshape(g.shape[0:2] + flat_corrs) for g in grids]

    return numba_grid(vis, uvw, flags, weights, ref_wave,
                      convolution_filter, w_bins, cell_size, grids)


@jit(nopython=True, nogil=True, cache=True)
def numba_degrid(grids, uvw, weights, ref_wave, convolution_filter,
                 w_bins, cell_size, vis):

    assert len(grids) == w_bins.shape[0] - 1
    bin_indices = np.digitize(uvw[:, 2], w_bins) - 1

    if np.any(bin_indices < 0):
        raise ValueError("bin_index < 0")

    if np.any(bin_indices >= len(grids)):
        raise ValueError("bin_index >= len(grids)")

    for i, grid in enumerate(grids):
        # The row mask for this layer
        mask = bin_indices == i
        rows = np.sum(mask)

        # Nothing to degrid
        if rows == 0:
            continue

        _, chan, corr = vis.shape

        res_vis = np.zeros((rows, chan, corr), dtype=grid.dtype)

        simple_numba_degrid(grid,
                            uvw[mask, ...],
                            weights[mask, ...],
                            ref_wave,
                            convolution_filter,
                            cell_size,
                            res_vis)

        vis[mask, ...] = res_vis

    return vis


def degrid(grids, uvw, weights, ref_wave,
           convolution_filter, w_bins, cell_size,
           dtype=np.complex64):
    """
    Convolutional W-stacking degridder (continuum)

    Variable numbers of correlations are supported.

    * :code:`(ny, nx, corr_1, corr_2)` ``grid`` will result in a
      :code:`(row, chan, corr_1, corr_2)` ``vis``

    * :code:`(ny, nx, corr_1)` ``grid`` will result in a
      :code:`(row, chan, corr_1)` ``vis``

    Parameters
    ----------
    grids : list of np.ndarray
        list of visibility grids of length :code:`nw`.
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
    w_bins : :class:`numpy.ndarray`
        W coordinate bins of shape :code:`(nw + 1,)`
    cell_size : float
        Cell size in arcseconds.
    dtype : :class:`numpy.dtype`, optional
        Numpy type of the resulting array. Defaults to
        :class:`numpy.complex64`.

    Returns
    -------
    np.ndarray
        :code:`(row, chan, corr_1, corr_2)` complex ndarray of visibilities
    """
    corrs = grids[0].shape[2:]
    nrow = uvw.shape[0]
    nchan = ref_wave.shape[0]

    # Flatten the correlation dimensions
    flat_corrs = reduce(mul, corrs)

    # Create output visibilities
    vis = np.empty((nrow, nchan, flat_corrs), dtype=dtype)

    grids = [g.reshape(g.shape[0:2] + (flat_corrs,)) for g in grids]

    numba_degrid(grids, uvw, weights, ref_wave, convolution_filter,
                 w_bins, cell_size, vis)

    return vis.reshape((nrow, nchan) + corrs)
