#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

from itertools import product

import numpy as np
import pytest

from africanus.constants import c as lightspeed
from africanus.gridding.util import estimate_cell_size


def rf(*a, **kw):
    return np.random.random(*a, **kw)


def test_degridder_gridder():
    """ Basic test of the gridder/degridder """
    from africanus.filters import convolution_filter
    from africanus.gridding.simple import grid, degrid

    conv_filter = convolution_filter("kaiser-bessel", 7, 7)
    nx = ny = npix = 257
    corr = (2, 2)
    chan = 4
    nvis = npix*npix
    npoints = 10000

    cell_size = 6  # 6 arc seconds

    # Channels of MeerKAT L band
    wavelengths = lightspeed/np.linspace(.856e9, .856e9*2, chan, endpoint=True)

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
    vis = degrid(sky_grid, uvw, weights, wavelengths,
                 conv_filter, cell_size,
                 dtype=sky_grid.dtype)

    assert vis.shape == (nvis, chan) + corr

    # Indicate all visibilities are unflagged
    flags = np.zeros_like(vis, dtype=np.bool)

    vis_grid = grid(vis, uvw, flags, weights, wavelengths,
                    conv_filter, cell_size, nx, ny)

    assert vis_grid.shape == (ny, nx) + corr

    # Test that a user supplied grid works
    vis_grid = grid(vis, uvw, flags, weights, wavelengths,
                    conv_filter, cell_size,
                    grid=vis_grid)

    assert vis_grid.shape == (ny, nx) + corr


@pytest.mark.parametrize("plot", [False])
@pytest.mark.parametrize("oversample", [
    1,
    pytest.param(2, marks=pytest.mark.xfail(reason="Unknown")),
    pytest.param(3, marks=pytest.mark.xfail(reason="Unknown")),
])
@pytest.mark.parametrize("even", [True, False])
def test_psf_subtraction(plot, oversample, even):
    """
    Test that we can create the PSF with the gridder.
    We do this by gridding vis and weights of one
    to create images of (ny, nx) and (ny*2, nx*2).
    Then we ensure that the centre of the second
    grid is equal to the entirety of the first
    """

    from africanus.filters import convolution_filter
    from africanus.gridding.simple import grid, degrid
    from numpy.fft import fft2, fftshift, ifft2, ifftshift

    np.random.seed(50)

    corr = (1,)
    chan = 16
    rows = 1024
    ny = nx = 512 if even else 513
    yodd = ny % 2
    xodd = nx % 2

    # Channels of MeerKAT L band
    wavelengths = lightspeed/np.linspace(.856e9, .856e9*2, chan, endpoint=True)

    # Random UVW coordinates
    uvw = np.empty((rows, 3), dtype=np.float64)
    uvw[:, :2] = (rf((rows, 2)) - 0.5)*100
    uvw[:, 2] = (rf((rows,)) - 0.5)*10

    # Estimate cell size given UVW coordinates and wavelengths
    cell_size = estimate_cell_size(uvw[:, 0], uvw[:, 1],
                                   wavelengths, factor=5).max()

    # We have the right number of pixels
    if even:
        assert ny % 2 == 0 and 2*(ny // 2) == ny
        assert nx % 2 == 0 and 2*(nx // 2) == nx
    else:
        assert ny % 2 == 1 and 2*(ny // 2) + 1 == ny
        assert nx % 2 == 1 and 2*(nx // 2) + 1 == nx

    assert ny == nx

    # Create image with a single point
    image = np.zeros((ny, nx), dtype=np.float64)
    image[ny // 2, nx // 2] = 1

    conv_filter = convolution_filter("kaiser-bessel", 7, oversample)
    weights = np.ones(shape=(rows, chan) + corr, dtype=np.float64)
    flags = np.zeros_like(weights, dtype=np.uint8)

    def centre_cut(img, cy, cx):
        return img[cy - cy // 2:cy + cy // 2 + img.shape[1] % 2,
                   cx - cx // 2:cx + cx // 2 + img.shape[0] % 2]

    # V = R(I)
    # Created a padded image,  FFT into the centre
    fft_image = np.zeros((2*ny + yodd, 2*nx + xodd), dtype=np.complex128)
    centre = centre_cut(fft_image, ny, nx)
    assert centre.shape == (ny, nx)
    centre[:] = fftshift(fft2(ifftshift(image)))

    assert np.sum(fft_image) == ny*nx
    assert np.any(fft_image != 0.0)

    vis = degrid(fft_image[:, :, None], uvw, weights, wavelengths,
                 conv_filter, 2*cell_size, dtype=np.complex128)

    assert vis.shape == (rows, chan, 1)
    assert np.any(vis != 0.0)
    assert vis.dtype == np.complex128

    # I^D = R+(V)
    grid_vis = np.zeros((2*ny + yodd, 2*nx + xodd, 1), dtype=np.complex128)
    centre = centre_cut(grid_vis, ny, nx)
    assert centre.shape == (ny, nx, 1)
    centre[:, :, :] = (grid(vis, uvw, flags, weights, wavelengths,
                            conv_filter, 2*cell_size, ny=ny, nx=nx))

    assert np.any(grid_vis != 0.0)

    dirty = fftshift(ifft2(ifftshift(grid_vis[:, :, 0]))).real

    assert grid_vis.dtype == np.complex128
    assert dirty.dtype == np.float64

    # PSF = R+(1)
    grid_unity = grid(np.ones_like(vis), uvw, flags, weights, wavelengths,
                      conv_filter, cell_size, ny=2*ny + yodd, nx=2*nx + xodd)

    psf = fftshift(ifft2(ifftshift(grid_unity[:, :, 0]))).real

    assert grid_unity.dtype == np.complex128
    assert psf.dtype == np.float64

    # Test that we have gridded something
    assert np.any(dirty != 0.0)
    assert np.any(psf != 0.0)

    norm_psf = psf / psf.max()
    norm_dirty = dirty / psf.max()

    psf, dirty = norm_psf, norm_dirty

    # Extract the centre of the PSF and the dirty image
    centre_psf = centre_cut(psf, ny, nx).copy()
    centre_dirty = centre_cut(dirty, ny, nx).copy()

    assert centre_psf.shape == centre_dirty.shape

    print(centre_dirty.max(), centre_psf.max())

    if plot:
        try:
            import matplotlib
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.fail("plotting requested but could not import matplotlib")

        plt.subplot(1, 4, 1)
        plt.imshow(centre_dirty, cmap="cubehelix")
        plt.title("CENTRE DIRTY")
        plt.colorbar()

        plt.subplot(1, 4, 2)
        plt.imshow(centre_psf, cmap="cubehelix")
        plt.title("CENTRE PSF")
        plt.colorbar()

        plt.subplot(1, 4, 3)
        plt.imshow(centre_psf - centre_dirty, cmap="cubehelix")
        plt.title("PSF - DIRTY")
        plt.colorbar()

        plt.subplot(1, 4, 4)
        plt.imshow(psf, cmap="cubehelix")
        plt.title("PSF")
        plt.colorbar()

        plt.show(True)

    # Must have some non-zero values
    assert np.any(dirty != 0.0)
    assert np.any(centre_psf != 0.0)

    # Should be very much the same
    assert np.allclose(centre_psf, centre_dirty, rtol=1e-64)


def test_dask_degridder_gridder():
    from africanus.filters import convolution_filter
    from africanus.gridding.simple.dask import grid, degrid

    da = pytest.importorskip('dask.array')

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
    wavelengths = lightspeed/da.linspace(.856e9, .856e9*2, chan,
                                         chunks=chan_chunk)
    flags = da.random.randint(0, 1, size=vis_shape, chunks=vis_chunks)

    weights = da.random.random(vis_shape, chunks=vis_chunks)

    conv_filter = convolution_filter("kaiser-bessel", 7, 7)

    vis_grid = grid(vis, uvw, flags, weights, wavelengths, conv_filter,
                    cell_size, ny, nx)

    degrid_vis = degrid(vis_grid, uvw, weights, wavelengths,
                        conv_filter, cell_size)

    np_vis_grid, np_degrid_vis = da.compute(vis_grid, degrid_vis)
    assert np_vis_grid.shape == (ny, nx) + corr
    assert np_degrid_vis.shape == (row, chan) + corr
