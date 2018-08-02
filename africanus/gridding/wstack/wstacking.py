from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import reduce
from operator import mul

import numba
import numpy as np

from africanus.gridding.simple.gridding import (
                numba_grid as simple_numba_grid)


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
        Minimum W coordinate
    w_max : float
        Maximum W coordinate
    l : :class:`numpy.ndarray`
        l coordinates
    m : :class:`numpy.ndarray`
        m coordinates

    Returns
    -------
    int
        Number of w-layers
    """
    max_n = np.sqrt(1 - l[None, :]**2 - m[:, None]**2).max()

    return np.ceil(2*np.pi*(w_max - w_min)/max_n).astype(np.int32).item()


def w_stacking_bins(w_min, w_max, w_layers):
    r"""
    Returns the W coordinate bins appropriate for the observation parameters,
    given the minimum and maximum W coordinates and the number of W layers.

    W coordinates can be binned by calling

    .. code-block:: python

        w_bins = np.digitize(w, bins) - 1

    Parameters
    ----------
    w_min : float
        Minimum W coordinate
    w_max : float
        Maximum W coordinate
    w_layers : int
        Number of w layers

    Returns
    -------
    :class:`numpy.ndarray`
        W-coordinate bins of shape :code:`(nw + 1,)`.
    """
    return np.linspace(w_min, w_max, w_layers + 1)


def w_stacking_centroids(w_bins):
    r"""
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
        W-coordinate centroids of shape :code:`(nw,)`.
    """
    return (w_bins[1:] + w_bins[:1]) / 2.0


@numba.jit(nopython=True, nogil=True, cache=True)
def w_bin_masks(uvw, w_bins):
    indices = np.digitize(uvw[:, 2], w_bins) - 1
    return [i == indices for i in range(w_bins.shape[0])]


@numba.jit(nopython=True, nogil=True, cache=True)
def numba_grid(vis, uvw, flags, weights, ref_wave,
               convolution_filter, w_bins, grids):
    assert len(grids) == w_bins.shape[0]
    bin_indices = np.digitize(uvw[:, 2], w_bins) - 1

    for i, grid in enumerate(grids):
        mask = bin_indices == i
        simple_numba_grid(vis[mask, ...],
                          uvw[mask, ...],
                          flags[mask, ...],
                          weights[mask, ...],
                          ref_wave,
                          convolution_filter,
                          grid)

    return grids


def grid(vis, uvw, flags, weights, ref_wave,
         convolution_filter, w_bins,
         nx=1024, ny=1024,
         grids=None):
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
        W coordinate bins of shape :code:`(nw,)`
    nx : integer, optional
        Size of the grid's X dimension
    ny : integer, optional
        Size of the grid's Y dimension
    grids : list of np.ndarray, optional
        list of complex64/complex128 arrays, each with shape
        :code:`(ny, nx, corr_1, corr_2)`
        If supplied, this array will be used as the gridding target,
        and ``nx`` and ``ny`` will be derived from the grid's
        dimensions.

    Returns
    -------
    list of np.ndarray
        :code:`(ny, nx, corr_1, corr_2)` complex ndarray of
        gridded visibilities. The number of correlations may vary,
        depending on the shape of vis.
    """
    corrs = vis.shape[2:]

    # Create grid of flatten correlations or reshape
    if grids is None:
        nw = w_bins.shape[0]
        grids = [np.zeros((ny, nx) + corrs, dtype=vis.dtype)
                 for _ in range(nw)]
    elif not isinstance(grids, list):
        grids = [grids]

    # Flatten the correlation dimensions
    flat_corrs = (reduce(mul, corrs),)
    grids = [g.reshape(g.shape[0:2] + flat_corrs) for g in grids]

    return numba_grid(vis, uvw, flags, weights, ref_wave,
                      convolution_filter, w_bins, grids)
