#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

from itertools import product

import pytest

from africanus.constants import c as lightspeed

def test_degridder_gridder():
    """ Basic test of the gridder/degridder """

    import numpy as np

    from africanus.filters import convolution_filter
    from africanus.gridding.simple import grid, degrid

    conv_filter = convolution_filter(3, 21, "kaiser-bessel")
    nx = ny = npix = 257
    corr = (2,2)
    chan = 4
    nvis = npix*npix
    npoints = 10000

    ARCSEC2RAD = 4.8481e-6
    DELTA_PIX = 6 * ARCSEC2RAD
    UV_SCALE = npix * DELTA_PIX

    rf = lambda *a, **kw: np.random.random(*a, **kw)

    # Channels of MeerKAT L band
    ref_wave = lightspeed/np.linspace(.856e9, .856e9*2, chan, endpoint=True)

    # Random UVW coordinates
    uvw = rf(size=(nvis,3)).astype(np.float64)*128*UV_SCALE

    # Simulate a sky model of npoints random point sources
    # by setting the I correlation
    sky_model = np.zeros(shape=(nx,ny)+corr, dtype=np.float64)
    x = np.random.randint(0, nx, npoints)
    y = np.random.randint(0, ny, npoints)
    sky_model[x,y,0,0] = rf(size=npoints)

    from numpy.fft import fftshift, fft2, ifftshift


    sky_grid = np.empty_like(sky_model, dtype=np.complex128
                        if sky_model.dtype == np.float64 else np.complex64)

    corr_products = product(*(range(c) for c in corr))
    for c1, c2 in corr_products:
        sky_grid[:,:,c1,c2] = fftshift(fft2(ifftshift(sky_model[:,:,c1,c2])))

    weights = np.random.random(size=(nvis,chan) + corr)

    # Degrid the sky model to produce visibilities
    vis = degrid(sky_grid, uvw, weights, ref_wave, conv_filter)

    assert vis.shape == (nvis, chan) + corr

    # Indicate all visibilities are unflagged
    flags = np.zeros_like(vis, dtype=np.bool)

    vis_grid = grid(vis, uvw, flags, weights, ref_wave,
                                        conv_filter, nx, ny)

    assert vis_grid.shape == (ny, nx) + corr

    # Test that a user supplied grid works
    vis_grid = grid(vis, uvw, flags, weights, ref_wave,
                                conv_filter, grid=vis_grid)

    assert vis_grid.shape == (ny, nx) + corr

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

    corr = (2,2)
    chan = 16
    rows = 200
    npix = ny = nx = 257

    ARCSEC2RAD = 4.8481e-6
    DELTA_PIX = 6 * ARCSEC2RAD
    UV_SCALE = npix * DELTA_PIX

    rf = lambda *a, **kw: np.random.random(*a, **kw)

    # Channels of MeerKAT L band
    ref_wave = lightspeed/np.linspace(.856e9, .856e9*2, chan, endpoint=True)

    # Random UVW coordinates
    uvw = rf(size=(rows,3)).astype(np.float64)*128*UV_SCALE

    # Visibilities and weight of one
    vis = np.ones(shape=(rows,chan) + corr, dtype=np.complex64)
    weights = np.ones(shape=(rows,chan) + corr, dtype=np.float32)

    # Indicate all visibilities are unflagged
    flags = np.zeros_like(vis, dtype=np.bool)

    conv_filter = convolution_filter(3, 21, "kaiser-bessel")

    # Compute PSF of (ny, nx)
    psf = grid(vis, uvw, flags, weights, ref_wave,
                                        conv_filter, nx, ny)

    # Compute PSF of (ny*2, nx*2)
    psf_squared = grid(vis, uvw, flags, weights, ref_wave,
                                        conv_filter, ny*2, nx*2)

    # Test that we have gridded something
    assert np.any(psf_squared > 0.0)
    assert np.any(psf > 0.0)

    # Extract the centre of the squared PSF
    centre_vis = psf_squared[ny-ny//2:1+ny+ny//2, nx-nx//2:1+nx+nx//2,:,:]

    # Should be the same
    assert np.all(centre_vis == psf)

from africanus.gridding.simple.dask import have_requirements

@pytest.mark.skipif(not have_requirements, reason="requirements not installed")
def test_dask_degridder_gridder():
    from africanus.filters import convolution_filter
    from africanus.gridding.simple.dask import grid, degrid

    import dask.array as da

    row = 100
    chan = 16
    corr = (2,2)
    nx = ny = 1024

    row_chunk = 25
    chan_chunk = 4
    corr_chunk = corr

    vis_shape = (row, chan) + corr
    vis_chunks = (row_chunk, chan_chunk) + corr_chunk

    vis = (da.random.random(vis_shape, chunks=vis_chunks) +
            1j*da.random.random(vis_shape, chunks=vis_chunks))
    uvw = da.random.random((row,3), chunks=(row_chunk, 3))
    # 4 channels of MeerKAT L band
    ref_wave = lightspeed/da.linspace(.856e9, .856e9*2, chan,
                            chunks=chan_chunk)
    flags = da.random.randint(0, 1, size=vis_shape, chunks=vis_chunks)

    weights = da.random.random(vis_shape, chunks=vis_chunks)

    conv_filter = convolution_filter(3, 21, "kaiser-bessel")

    vis_grid = grid(vis, uvw, flags, weights, ref_wave, conv_filter, ny, nx)

    degrid_vis = degrid(vis_grid, uvw, weights, ref_wave, conv_filter)

    np_vis_grid, np_degrid_vis = da.compute(vis_grid, degrid_vis)
    assert np_vis_grid.shape == (ny, nx) + corr
    assert np_degrid_vis.shape == (row, chan) + corr



