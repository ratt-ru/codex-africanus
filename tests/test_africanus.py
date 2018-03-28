#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

import pytest

def test_degridder_gridder():
    """ Basic test of the gridder/degridder """

    import numpy as np

    from africanus.filters import convolution_filter
    from africanus.gridding.simple import grid, psf, degrid

    conv_filter = convolution_filter(3, 63, "sinc")
    nx = ny = npix = 257
    ncorr = 1
    nchan = 4
    nvis = npix*npix
    npoints = 10000

    C = 2.99792458e8
    ARCSEC2RAD = 4.8481e-6
    DELTA_PIX = 6 * ARCSEC2RAD
    UV_SCALE = npix * DELTA_PIX

    rf = lambda *a, **kw: np.random.random(*a, **kw)

    # 4 channels of MeerKAT L band
    ref_wave = C/np.linspace(.856e9, .856e9*2, nchan, endpoint=True)

    # Random UVW coordinates
    uvw = rf(size=(nvis,3)).astype(np.float64)*128*UV_SCALE

    # Simulate a sky model of npoints random point sources
    sky_model = np.zeros(shape=(ncorr,nx,ny), dtype=np.float64)
    x = np.random.randint(0, nx, npoints)
    y = np.random.randint(0, ny, npoints)
    sky_model[0,x,y] = rf(size=npoints)

    from numpy.fft import fftshift, fft2, ifftshift

    sky_grid = np.stack([fftshift(fft2(ifftshift(sky_model[c])))
                                    for c in range(ncorr)],
                                                    axis=0)

    # Degrid the sky model to produce visibilities
    vis = degrid(sky_grid, uvw, ref_wave, conv_filter)


    # Indicate all visibilities are unflagged
    flags = np.zeros_like(vis, dtype=np.bool)

    vis_grid = grid(vis, uvw, flags, ref_wave, nx, ny, conv_filter)
    psf_ = psf(vis, uvw, flags, ref_wave, nx, ny, conv_filter)
