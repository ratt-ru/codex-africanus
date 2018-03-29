#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

import pytest

def test_degridder_gridder():
    """ Basic test of the gridder/degridder """

    import numpy as np

    from africanus.filters import convolution_filter
    from africanus.gridding.simple import grid, degrid

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

    # Channels of MeerKAT L band
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

    weights = np.random.random(size=(nvis,nchan,ncorr))

    # Degrid the sky model to produce visibilities
    vis = degrid(sky_grid, uvw, weights, ref_wave, conv_filter)


    # Indicate all visibilities are unflagged
    flags = np.zeros_like(vis, dtype=np.bool)

    vis_grid = grid(vis, uvw, flags, weights, ref_wave,
                                        conv_filter, nx, ny)

def test_psf_subtraction():
    """
    Test that we can create the PSF with the gridder.
    We do this by gridding vis and weights of one
    to create images of (ny, nx) and (ny*2, nx*2).
    Then we ensure that the centre of the second
    grid is equal to the entirety of the first
    """

    from africanus.filters import convolution_filter
    from africanus.gridding.simple import grid, degrid
    import numpy as np

    corr = 4
    chan = 16
    rows = 200
    npix = nx = ny = 257

    C = 2.99792458e8
    ARCSEC2RAD = 4.8481e-6
    DELTA_PIX = 6 * ARCSEC2RAD
    UV_SCALE = npix * DELTA_PIX

    rf = lambda *a, **kw: np.random.random(*a, **kw)

    # Channels of MeerKAT L band
    ref_wave = C/np.linspace(.856e9, .856e9*2, chan, endpoint=True)

    # Random UVW coordinates
    uvw = rf(size=(rows,3)).astype(np.float64)*128*UV_SCALE

    # Visibilities and weight of one
    vis = np.ones(shape=(rows,chan,corr), dtype=np.complex64)
    weights = np.ones(shape=(rows,chan,corr), dtype=np.float32)

    # Indicate all visibilities are unflagged
    flags = np.zeros_like(vis, dtype=np.bool)

    conv_filter = convolution_filter(3, 63, "sinc")

    # Compute PSF of (ny, nx)
    psf = grid(vis, uvw, flags, weights, ref_wave,
                                        conv_filter, nx, ny)

    # Compute PSF of (ny*2, nx*2)
    psf_squared = grid(vis, uvw, flags, weights, ref_wave,
                                        conv_filter, nx*2, ny*2)

    # Test that we have gridded something
    assert np.any(psf_squared > 0.0)
    assert np.any(psf > 0.0)

    # Extract the centre of the squared PSF
    centre_vis = psf_squared[:,nx-nx//2:1+nx+nx//2, nx-nx//2:1+nx+nx//2]

    # Should be the same
    assert np.all(centre_vis == psf)





