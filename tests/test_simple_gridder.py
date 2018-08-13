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

    conv_filter = convolution_filter(3, 63, "sinc")
    nx = ny = npix = 257
    corr = (2, 2)
    chan = 4
    nvis = npix*npix
    npoints = 10000

    cell_size = 6  # 6 arc seconds

    rf = lambda *a, **kw: np.random.random(*a, **kw)

    # Channels of MeerKAT L band
    ref_wave = lightspeed/np.linspace(.856e9, .856e9*2, chan, endpoint=True)

    # Random UVW coordinates
    uvw = rf(size=(nvis, 3)).astype(np.float64)*128

    # Simulate a sky model of npoints random point sources
    # by setting the I correlation
    sky_model = np.zeros(shape=(nx, ny)+corr, dtype=np.float64)
    x = np.random.randint(0, nx, npoints)
    y = np.random.randint(0, ny, npoints)
    sky_model[x, y, 0, 0] = rf(size=npoints)

    from numpy.fft import fftshift, fft2, ifftshift

    sky_grid = np.empty_like(sky_model, dtype=np.complex128
                             if sky_model.dtype == np.float64
                             else np.complex64)

    corr_products = product(*(range(c) for c in corr))
    for c1, c2 in corr_products:
        transform = ifftshift(sky_model[:, :, c1, c2])
        sky_grid[:, :, c1, c2] = fftshift(fft2(transform))

    weights = np.random.random(size=(nvis, chan) + corr)

    # Degrid the sky model to produce visibilities
    vis = degrid(sky_grid, uvw, weights, ref_wave, conv_filter, cell_size)

    assert vis.shape == (nvis, chan) + corr

    # Indicate all visibilities are unflagged
    flags = np.zeros_like(vis, dtype=np.bool)

    vis_grid = grid(vis, uvw, flags, weights, ref_wave,
                    conv_filter, cell_size, nx, ny)

    assert vis_grid.shape == (ny, nx) + corr

    # Test that a user supplied grid works
    vis_grid = grid(vis, uvw, flags, weights, ref_wave,
                    conv_filter, cell_size,
                    grid=vis_grid)

    assert vis_grid.shape == (ny, nx) + corr


@pytest.mark.skip
def test_psf_subtraction():
    """
    Test that we can create the PSF with the gridder.
    We do this by gridding vis and weights of one
    to create images of (ny, nx) and (ny*2, nx*2).
    Then we ensure that the centre of the second
    grid is equal to the entirety of the first
    """

    from africanus.filters import convolution_filter
    from africanus.gridding.simple import grid
    import numpy as np

    corr = (2, 2)
    chan = 16
    rows = 200
    ny = nx = 17

    cell_size = 6  # 6 arcseconds

    rf = lambda *a, **kw: np.random.random(*a, **kw)

    # Channels of MeerKAT L band
    ref_wave = lightspeed/np.linspace(.856e9, .856e9*2, chan, endpoint=True)

    # Random UVW coordinates
    uvw = rf(size=(rows, 3)).astype(np.float64)*128

    # Visibilities and weight of one
    vis = np.ones(shape=(rows, chan) + corr, dtype=np.complex64)
    weights = np.ones(shape=(rows, chan) + corr, dtype=np.float32)

    # Indicate all visibilities are unflagged
    flags = np.zeros_like(vis, dtype=np.bool)

    conv_filter = convolution_filter(3, 63, "sinc")

    # Compute PSF of (ny, nx)
    psf = grid(vis, uvw, flags, weights, ref_wave,
               conv_filter, cell_size, nx, ny)

    # Compute PSF of (ny*2, nx*2)
    psf_squared = grid(vis, uvw, flags, weights, ref_wave,
                       conv_filter, cell_size, ny*2, nx*2)

    # Test that we have gridded something
    assert np.any(psf_squared > 0.0)
    assert np.any(psf > 0.0)

    # Extract the centre of the squared PSF
    centre_vis = psf_squared[ny-ny//2:1+ny+ny//2, nx-nx//2:1+nx+nx//2, :, :]

    assert np.any(centre_vis > 0.0)

    # Should be the same
    assert np.all(centre_vis == psf)


def test_psf_subtraction2():
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
    from numpy.fft import fft2, fftshift, ifft2, ifftshift

    np.random.seed(50)

    corr = (1,)
    chan = 16
    rows = 200
    ny = nx = 513

    cell_size = 6  # 6 arcseconds

    rf = lambda *a, **kw: np.random.random(*a, **kw)

    # Channels of MeerKAT L band
    ref_wave = lightspeed/np.linspace(.856e9, .856e9*2, chan, endpoint=True)

    # Random UVW coordinates
    uvw = np.empty((rows, 3), dtype=np.float64)
    uvw[:, :2] = np.random.random((rows, 2))*64
    uvw[:, 2] = np.random.random(rows)*5

    umin, umax = uvw[:, 0].min(), uvw[:, 0].max()
    vmin, vmax = uvw[:, 1].min(), uvw[:, 1].max()

    umin *= ref_wave[-1]
    umax *= ref_wave[-1]

    vmin *= ref_wave[-1]
    vmax *= ref_wave[-1]

    _ARCSEC2RAD = np.deg2rad(1.0/(60*60))

    assert cell_size*_ARCSEC2RAD < 1.0 / (2*umax)
    assert cell_size*_ARCSEC2RAD < 1.0 / (2*vmax)

    # assert cell_size*_ARCSEC2RAD*nx >= (1.0 / np.abs(vmin))
    # assert cell_size*_ARCSEC2RAD*ny >= (1.0 / np.abs(umin))

    image = np.zeros((ny, nx), dtype=np.float64)

    # We have the centre pixel
    assert ny % 2 == 1 and 2*(ny // 2) + 1 == ny
    assert nx % 2 == 1 and 2*(nx // 2) + 1 == nx

    image[ny // 2, nx // 2] = 1

    conv_filter = convolution_filter(3, 7, "sinc")
    weights = np.ones(shape=(rows, chan) + corr, dtype=np.float64)
    flags = np.zeros_like(weights, dtype=np.uint8)

    # V = R(I)
    # fft_image = ifftshift(fft2(fftshift(image), norm="ortho"))
    fft_image = fftshift(fft2(ifftshift(image)))

    # I^D = R+(V)
    vis = degrid(fft_image[:, :, None], uvw, weights, ref_wave,
                 conv_filter, cell_size)

    assert vis.shape == (rows, chan, 1)

    grid_vis = grid(vis, uvw, flags, weights, ref_wave,
                    conv_filter, cell_size, ny=ny, nx=nx)

    # dirty = fftshift(ifft2(ifftshift(grid_vis[:, :, 0]), norm="ortho")).real
    dirty = fftshift(ifft2(ifftshift(grid_vis[:, :, 0]))).real

    # PSF = R+(1)
    conv_filter = convolution_filter(3, 15, "sinc")
    grid_unity = grid(np.ones_like(vis), uvw, flags, weights, ref_wave,
                      conv_filter, cell_size, ny=2*ny + 1, nx=2*nx + 1)

    # psf = fftshift(ifft2(ifftshift(grid_unity[:, :, 0]), norm="ortho")).real
    psf = fftshift(ifft2(ifftshift(grid_unity[:, :, 0]))).real

    # Test that we have gridded something
    assert np.any(dirty != 0.0)
    assert np.any(psf != 0.0)

    # Extract the centre of the squared PSF
    centre_psf = psf[ny - ny//2:1 + ny + ny//2, nx - nx//2:1 + nx + nx//2].copy()

    #centre_psf = psf[ny - ny//2 - 1:ny + ny//2, nx - nx//2 - 1:nx + nx//2].copy()

    # Normalise by size
    centre_psf *= 4

    assert centre_psf.shape == dirty.shape

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pass
    else:

        plt.subplot(1, 4, 1)
        plt.imshow(dirty, cmap="cubehelix")
        plt.title("DIRTY")
        plt.colorbar()

        plt.subplot(1, 4, 2)
        plt.imshow(centre_psf, cmap="cubehelix")
        plt.title("CENTRE PSF")
        plt.colorbar()

        plt.subplot(1, 4, 3)
        plt.imshow(dirty - centre_psf, cmap="cubehelix")
        plt.title("DIRTY - CENTRE PSF")
        plt.colorbar()

        plt.subplot(1, 4, 4)
        plt.imshow(psf, cmap="cubehelix")
        plt.title("PSF")
        plt.colorbar()

        plt.show(True)

    assert np.any(centre_psf != 0.0)

    # Should be the same
    assert np.all(centre_psf == dirty)


from africanus.gridding.simple.dask import have_requirements


@pytest.mark.skipif(not have_requirements, reason="requirements not installed")
def test_dask_degridder_gridder():
    from africanus.filters import convolution_filter
    from africanus.gridding.simple.dask import grid, degrid

    import dask.array as da

    row = 100
    chan = 16
    corr = (2, 2)
    nx = ny = 1024
    cell_size = 6

    row_chunk = 25
    chan_chunk = 4
    corr_chunk = corr

    vis_shape = (row, chan) + corr
    vis_chunks = (row_chunk, chan_chunk) + corr_chunk

    vis = (da.random.random(vis_shape, chunks=vis_chunks) +
           1j*da.random.random(vis_shape, chunks=vis_chunks))
    uvw = da.random.random((row, 3), chunks=(row_chunk, 3))
    # 4 channels of MeerKAT L band
    ref_wave = lightspeed/da.linspace(.856e9, .856e9*2, chan,
                                      chunks=chan_chunk)
    flags = da.random.randint(0, 1, size=vis_shape, chunks=vis_chunks)

    weights = da.random.random(vis_shape, chunks=vis_chunks)

    conv_filter = convolution_filter(3, 63, "sinc")

    vis_grid = grid(vis, uvw, flags, weights, ref_wave, conv_filter,
                    cell_size, ny, nx)

    degrid_vis = degrid(vis_grid, uvw, weights, ref_wave,
                        conv_filter, cell_size)

    np_vis_grid, np_degrid_vis = da.compute(vis_grid, degrid_vis)
    assert np_vis_grid.shape == (ny, nx) + corr
    assert np_degrid_vis.shape == (row, chan) + corr
