# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numba
import numpy as np

def _grid(vis, uvw, flags, ref_wave, nx, ny, convolution_filter):
    """
    Convolutional gridder (continuum)

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
    nx : integer
        Size of the grid's X dimension
    ny : integer
        Size of the grid's Y dimension
    convolution_filter :  :class:`AAFilter`
        Anti-Aliasing filter object

    Returns
    -------
    np.ndarray
        (corr, nx, ny) of complex64 gridded visibilities
    """
    cf = convolution_filter

    assert vis.shape[1] == ref_wave.shape[0]
    filter_index = np.arange(-cf.half_sup, cf.half_sup+1)

    # one grid for the resampled visibilities per correlation:
    grid = np.zeros((vis.shape[2],ny,nx),dtype=np.complex64)

    # 1. Foreach row in block
    #    (A) apply beam DDE gains
    #        Sample the facet centre position
    #        in the beam cube as it varies over time
    #        to produce these gains.
    #    (B) average
    # 2. Foreach row in averaged block

    for r in range(uvw.shape[0]):                 # row (vis)
        for c in range(vis.shape[1]):             # channel

            scaled_u = uvw[r,0] / ref_wave[c]
            scaled_v = uvw[r,1] / ref_wave[c]

            disc_u = int(round(scaled_u))
            disc_v = int(round(scaled_v))

            frac_u = int((1 + cf.half_sup +
                         (-scaled_u + disc_u)) * cf.oversample)
            frac_v = int((1 + cf.half_sup +
                         (-scaled_v + disc_v)) * cf.oversample)

            # Out of bounds check
            if (disc_v + ny // 2 + cf.half_sup >= ny or
                disc_u + nx // 2 + cf.half_sup >= nx or
                disc_v + ny // 2 - cf.half_sup < 0 or
                disc_u + nx // 2 - cf.half_sup < 0):
                continue

            for conv_v in filter_index:
                v_tap = cf.filter_taps[conv_v*cf.oversample + frac_v]
                grid_v = disc_v + conv_v + ny // 2

                for conv_u in filter_index:
                    u_tap = cf.filter_taps[conv_u*cf.oversample + frac_u]
                    conv_weight = v_tap * u_tap
                    grid_u = disc_u + conv_u + nx // 2

                    for p in range(vis.shape[2]): # polarisation/correlation
                        # Ignore flagged visibilities
                        if flags[r, c, p] > 0:
                            continue

                        grid[p,grid_v,grid_u] += (vis[r,c,p]*conv_weight)

    return grid


def _psf(vis, uvw, flags, ref_wave, nx, ny, convolution_filter):
    """
    Convolutional PSF gridder (continuum)

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
    nx : integer
        Size of the grid's X dimension
    ny : integer
        Size of the grid's Y dimension
    convolution_filter :  :class:`AAFilter`
        Anti-Aliasing filter object

    Returns
    -------
    np.ndarray
        (nx, ny) containing the PSF, constructed
        by summing convolutional weights.
    """
    cf = convolution_filter

    assert vis.shape[1] == ref_wave.shape[0]
    filter_index = np.arange(-cf.half_sup, cf.half_sup+1)

    # for deconvolution the PSF should be 2x size of the image (see
    # Hogbom CLEAN for details), one grid for the sampling function:
    grid = np.zeros((2*ny,2*nx), dtype=np.complex64)
    rgrid = grid.real

    for r in range(uvw.shape[0]):                 # row (vis)
        for c in range(vis.shape[1]):             # channel
            # Continue if any correlation is flagged
            flagged = False

            for p in range(vis.shape[2]):
                if flags[r,c,p] > 0:
                    flagged = True
                    break

            if flagged:
                continue

            scaled_v = uvw[r,0] / ref_wave[c]
            scaled_u = uvw[r,1] / ref_wave[c]

            disc_u = int(round(scaled_u))
            disc_v = int(round(scaled_v))

            # Out of bounds check, this mirrors the code
            # in grid_vis and could probably be redone in
            # terms of disc_u_psf and disc_v_psf below
            if (disc_v + ny // 2 + cf.half_sup >= ny or
                disc_u + nx // 2 + cf.half_sup >= nx or
                disc_v + ny // 2 - cf.half_sup < 0 or
                disc_u + nx // 2 - cf.half_sup < 0):
                continue

            disc_u_psf = int(round(scaled_u*2))
            disc_v_psf = int(round(scaled_v*2))

            frac_u = int((1 + cf.half_sup + disc_u_psf - scaled_u*2)
                                                    * cf.oversample)
            frac_v = int((1 + cf.half_sup + disc_v_psf - scaled_v*2)
                                                    * cf.oversample)

            for conv_v in filter_index:
                v_tap_psf = cf.filter_taps[conv_v*cf.oversample + frac_v]
                grid_v = disc_v_psf + conv_v + ny

                for conv_u in filter_index:
                    u_tap_psf = cf.filter_taps[conv_u*cf.oversample + frac_u]
                    conv_weight_psf = v_tap_psf * u_tap_psf
                    grid_u = disc_u_psf + conv_u + nx

                    # PSF imaginary numbers are zero
                    rgrid[grid_v, grid_u] += conv_weight_psf

    return grid

def _degrid(grid, uvw, ref_wave, convolution_filter):
    """
    Convolutional degridder (continuum)

    Parameters
    ----------
    grid : np.ndarray
        float or complex grid of visibilities
        of shape (ncorr, nx, ny)
    uvw : np.ndarray
        float64 array of UVW coordinates of shape (row, 3)
    ref_wave : np.ndarray
        float64 array of wavelengths of shape (chan,)
    convolution_filter :  :class:`AAFilter`
        Anti-Aliasing filter object

    Returns
    -------
    np.ndarray
        (nrow, nchan, ncorr) array
        of complex64 gridded visibilities
    """
    cf = convolution_filter
    ncorr, nx, ny = grid.shape
    nchan = ref_wave.shape[0]
    nrow = uvw.shape[0]

    vis = np.zeros((nrow, nchan, ncorr), dtype=np.complex64)

    assert vis.shape[1] == ref_wave.shape[0]
    filter_index = np.arange(-cf.half_sup, cf.half_sup+1)

    for r in range(uvw.shape[0]):                 # row (vis)
        for c in range(vis.shape[1]):             # channel

            scaled_u = uvw[r,0] / ref_wave[c]
            scaled_v = uvw[r,1] / ref_wave[c]

            disc_u = int(round(scaled_u))
            disc_v = int(round(scaled_v))

            frac_u = int((1 + cf.half_sup +
                         (-scaled_u + disc_u)) * cf.oversample)
            frac_v = int((1 + cf.half_sup +
                         (-scaled_v + disc_v)) * cf.oversample)

            # Out of bounds check
            if (disc_v + ny // 2 + cf.half_sup >= ny or
                disc_u + nx // 2 + cf.half_sup >= nx or
                disc_v + ny // 2 - cf.half_sup < 0 or
                disc_u + nx // 2 - cf.half_sup < 0):
                continue

            for conv_v in filter_index:
                v_tap = cf.filter_taps[conv_v*cf.oversample + frac_v]
                grid_v = disc_v + conv_v + ny // 2

                for conv_u in filter_index:
                    u_tap = cf.filter_taps[conv_u*cf.oversample + frac_u]
                    conv_weight = v_tap * u_tap
                    grid_u = disc_u + conv_u + nx // 2

                    for p in range(vis.shape[2]): # polarisation/correlation
                        vis[r,c,p] += grid[p,grid_v,grid_u] * conv_weight

    return vis

# jit the functions if this is not RTD
import os

if os.environ.get('READTHEDOCS') == 'True':
    grid = _grid
    degrid = _degrid
    psf = _psf
else:
    grid = numba.jit(nopython=True, nogil=True, cache=True)(_grid)
    degrid = numba.jit(nopython=True, nogil=True, cache=True)(_degrid)
    psf = numba.jit(nopython=True, nogil=True, cache=True)(_psf)
