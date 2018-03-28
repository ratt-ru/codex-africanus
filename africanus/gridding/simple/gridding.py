# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numba
import numpy as np

def _grid(vis, uvw, flags, ref_wave,
                convolution_filter,
                nx=1024, ny=1024):
    """
    Convolutional gridder which grids visibilities ``vis``
    at the specified ``uvw`` coordinates and
    ``ref_wave`` reference wavelengths using
    the specified ``convolution_filter``.

    It returns both the **gridded visibilities** and
    **point spread function** as complex64 arrays.

    Parameters
    ----------
    vis : np.ndarray
        complex64 visibility array of shape (row, chan, corr)
    uvw : np.ndarray
        float64 array of UVW coordinates of shape (row, 3)
    flags : np.ndarray
        flagged data array of shape (row, chan, corr).
        Any positive quantity will indicate that the corresponding
        visibility should be flagged
    ref_wave : np.ndarray
        float64 array of wavelengths of shape (chan,)
    convolution_filter :  :class:`~africanus.filters.ConvolutionFilter`
        Convolution filter
    nx : integer, optional
        Size of the grid's X dimension
    ny : integer, optional
        Size of the grid's Y dimension

    Returns
    -------
    np.ndarray
        (corr, ny, nx) complex64 ndarray of gridded visibilities
    np.ndarray
        (ny, nx) complex64 ndarray containing the PSF,
        constructed by summing convolutional weights.
    """
    cf = convolution_filter

    assert vis.shape[1] == ref_wave.shape[0]
    filter_index = np.arange(-cf.half_sup, cf.half_sup+1)

    # one grid for the resampled visibilities per correlation:
    grid = np.zeros((vis.shape[2],ny,nx),dtype=np.complex64)

    # for deconvolution the PSF should be 2x size of the image (see
    # Hogbom CLEAN for details), one grid for the sampling function:
    # Reference the real component of the PSF, imaginary is zero
    psf = np.zeros((2*ny,2*nx), dtype=np.complex64)
    psf_real = psf.real

    for r in range(uvw.shape[0]):                 # row (vis)
        for f in range(vis.shape[1]):             # channel (freq)
            scaled_u = uvw[r,0] / ref_wave[f]
            scaled_v = uvw[r,1] / ref_wave[f]

            disc_u = int(np.round(scaled_u))
            disc_v = int(np.round(scaled_v))

            # Out of bounds check
            if (disc_v + ny // 2 + cf.half_sup >= ny or
                disc_u + nx // 2 + cf.half_sup >= nx or
                disc_v + ny // 2 - cf.half_sup < 0 or
                disc_u + nx // 2 - cf.half_sup < 0):
                continue

            # One plus half support
            one_half_sup = 1 + cf.half_sup

            # Compute fractional u and v
            base_frac_u = one_half_sup + disc_u - scaled_u
            base_frac_v = one_half_sup + disc_v - scaled_v

            frac_u = int(base_frac_u*cf.oversample)
            frac_v = int(base_frac_v*cf.oversample)

            # Twice scaled u and v
            twice_scaled_u = 2*scaled_u
            twice_scaled_v = 2*scaled_u

            # Compute fractional u and v for the PSF
            disc_u_psf = int(np.round(twice_scaled_u))
            disc_v_psf = int(np.round(twice_scaled_v))

            base_frac_u_psf = one_half_sup + disc_u_psf - twice_scaled_u
            base_frac_v_psf = one_half_sup + disc_v_psf - twice_scaled_v

            frac_u_psf = int(base_frac_u_psf*cf.oversample)
            frac_v_psf = int(base_frac_v_psf*cf.oversample)

            # Iterate over v/y
            for conv_v in filter_index:
                base_v = conv_v*cf.oversample
                v_tap = cf.filter_taps[base_v + frac_v]
                v_tap_psf = cf.filter_taps[base_v + frac_v_psf]

                grid_v = disc_v + conv_v + ny // 2
                grid_v_psf = disc_v_psf + conv_v + ny

                # Iterate over u/x
                for conv_u in filter_index:
                    base_u = conv_u*cf.oversample
                    u_tap = cf.filter_taps[base_u + frac_u]
                    u_tap_psf = cf.filter_taps[base_u + frac_u_psf]

                    conv_weight = v_tap * u_tap
                    conv_weight_psf = v_tap_psf * u_tap_psf

                    grid_u = disc_u + conv_u + nx // 2
                    grid_u_psf = disc_u_psf + conv_u + nx

                    vis_flagged = False

                    for c in range(vis.shape[2]):      # correlation
                        # Ignore flagged correlations and
                        # indicate flagging within visibility
                        if flags[r,f,c] > 0:
                            vis_flagged = True
                            continue

                        # Grid the visibility
                        grid[c,grid_v,grid_u] +=  vis[r,f,c]*conv_weight

                    # Don't grid the PSF is *any* correlation is flagged
                    if vis_flagged:
                        continue

                    # Grid weight to PSF, assuming its the same
                    # for all correlations
                    psf_real[grid_v_psf,grid_u_psf] += conv_weight_psf

    return grid, psf

def _degrid(grid, uvw, ref_wave, convolution_filter):
    """
    Convolutional degridder (continuum)

    Parameters
    ----------
    grid : np.ndarray
        float or complex grid of visibilities
        of shape (corr, ny, nx)
    uvw : np.ndarray
        float64 array of UVW coordinates of shape (row, 3)
    ref_wave : np.ndarray
        float64 array of wavelengths of shape (chan,)
    convolution_filter :  :class:`~africanus.filters.ConvolutionFilter`
        Convolution Filter

    Returns
    -------
    np.ndarray
        (row, chan, corr) complex64 ndarray of visibilities
    """
    cf = convolution_filter
    ncorr, nx, ny = grid.shape
    nchan = ref_wave.shape[0]
    nrow = uvw.shape[0]

    vis = np.zeros((nrow, nchan, ncorr), dtype=np.complex64)

    assert vis.shape[1] == ref_wave.shape[0]
    filter_index = np.arange(-cf.half_sup, cf.half_sup+1)

    for r in range(uvw.shape[0]):                 # row (vis)
        for f in range(vis.shape[1]):             # channel

            scaled_u = uvw[r,0] / ref_wave[f]
            scaled_v = uvw[r,1] / ref_wave[f]

            disc_u = int(round(scaled_u))
            disc_v = int(round(scaled_v))

            # Out of bounds check
            if (disc_v + ny // 2 + cf.half_sup >= ny or
                disc_u + nx // 2 + cf.half_sup >= nx or
                disc_v + ny // 2 - cf.half_sup < 0 or
                disc_u + nx // 2 - cf.half_sup < 0):
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
                    for c in range(vis.shape[2]):
                        vis[r,f,c] += grid[c,grid_v,grid_u]*conv_weight

    return vis

# jit the functions if this is not RTD
import os

if os.environ.get('READTHEDOCS') == 'True':
    grid = _grid
    degrid = _degrid
else:
    grid = numba.jit(nopython=True, nogil=True, cache=True)(_grid)
    degrid = numba.jit(nopython=True, nogil=True, cache=True)(_degrid)
