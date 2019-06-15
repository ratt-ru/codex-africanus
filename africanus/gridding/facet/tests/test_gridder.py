# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from africanus.constants import c as lightspeed
from africanus.gridding.facet.gridding import grid, degrid
from africanus.gridding.facet.spheroidal import wplanes
from africanus.gridding.util import estimate_cell_size

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest


def rf(*args, **kwargs):
    return np.random.random(*args, **kwargs)


def rc(*args, **kwargs):
    return rf(*args, **kwargs) + 1j*rf(*args, **kwargs)


@pytest.mark.parametrize("support", [11])
@pytest.mark.parametrize("spheroidal_support", [111])
@pytest.mark.parametrize("npix", [1025])
@pytest.mark.parametrize("wlayers", [7])
@pytest.mark.parametrize("maxw", [30000])
@pytest.mark.parametrize("cell_size", [1.3])
@pytest.mark.parametrize("oversampling", [11])
@pytest.mark.parametrize("lm", [(1e-8, 1e-8)])
def test_degridder(support, spheroidal_support, npix,
                   wlayers, maxw, cell_size,
                   oversampling, lm):

    np.random.seed(42)
    nrow = 10
    nchan = 8
    ncorr = 4

    vis = rc((nrow, nchan, ncorr)).astype(np.complex64)
    uvw = rf((nrow, 3)) - 0.5
    uvw[:, :2] *= 1e4
    freqs = np.linspace(.856e9, 2*.856e9, nchan)
    ref_wave = freqs[nchan // 2] / lightspeed

    meta = wplanes(wlayers, cell_size, support, maxw,
                   npix, oversampling,
                   lm, freqs)

    grid_ = rc((npix, npix, ncorr)).astype(np.complex128)

    vis = degrid(grid_, uvw, freqs, meta)
    assert vis.shape == (uvw.shape[0], freqs.shape[0], grid_.shape[2])


@pytest.mark.parametrize("support", [11])
@pytest.mark.parametrize("spheroidal_support", [111])
@pytest.mark.parametrize("npix", [1025])
@pytest.mark.parametrize("wlayers", [7])
@pytest.mark.parametrize("maxw", [30000])
@pytest.mark.parametrize("cell_size", [1.3])
@pytest.mark.parametrize("oversampling", [11])
@pytest.mark.parametrize("lm", [(0.0, 0.0)])
def test_gridder(support, spheroidal_support, npix,
                 wlayers, maxw, cell_size,
                 oversampling, lm):

    np.random.seed(42)
    nrow = 10
    nchan = 8
    ncorr = 4

    uvw = rf((nrow, 3)) - 0.5
    uvw[:, :2] *= 1e4
    vis = rc((nrow, nchan, ncorr))
    flags = np.random.randint(0, 2, (nrow, nchan, ncorr))
    weights = np.random.random((nrow, nchan, ncorr))
    freqs = np.linspace(.856e9, 2*.856e9, nchan)

    ref_wave = freqs[nchan // 2] / lightspeed

    meta = wplanes(wlayers, cell_size, support, maxw,
                   npix, oversampling,
                   lm, freqs)

    grid_ = grid(vis, uvw, flags, weights, freqs, meta, ny=npix, nx=npix)

    assert grid_.shape == (npix, npix, ncorr)


@pytest.mark.parametrize("support", [11])
@pytest.mark.parametrize("npix", [513])
@pytest.mark.parametrize("wlayers", [7])
@pytest.mark.parametrize("oversampling", [11])
@pytest.mark.parametrize("lm", [(1e-8, 1e-8)])
@pytest.mark.parametrize("plot", [True])
def test_psf_subtraction(support, npix, wlayers,
                         oversampling, lm,
                         plot):

    def _kernel_meta(cell_size, npix):
        return wplanes(wlayers, cell_size, support, maxw,
                       npix, oversampling,
                       lm, freqs)

    from numpy.fft import fftshift, fft2, ifftshift, ifft2

    np.random.seed(42)

    nrow = 10
    nchan = 8
    ncorr = 1

    freqs = np.linspace(.856e9, 2*.856e9, nchan)
    wavelengths = lightspeed / freqs

    # Random UVW coordinates
    uvw = np.empty((nrow, 3), dtype=np.float64)
    uvw[:, :2] = (rf((nrow, 2)) - 0.5)*10000
    uvw[:, 2] = (rf((nrow,)) - 0.5)*10

    maxw = uvw[:, 2].max()

    # Estimate cell size given UVW coordiantes and wavelengths
    cell_size = estimate_cell_size(uvw[:, 0], uvw[:, 1],
                                   wavelengths, factor=3).max()

    assert npix % 2 == 1

    # Create image with point at centre
    image = np.zeros((npix, npix), dtype=np.float64)
    image[npix // 2, npix // 2] = 1

    weights = np.ones(shape=(nrow, nchan, ncorr), dtype=np.float64)
    flags = np.zeros_like(weights, dtype=np.uint8)

    npix_psf = 2*npix - 1
    npad_psf = (npix_psf - npix) // 2
    unpad = slice(npad_psf, -npad_psf)

    # V = R(I)
    # Pad the  image, FFT
    padding = ((npad_psf, npad_psf), (npad_psf, npad_psf))
    pad_image = np.pad(image, padding, mode='constant',
                       constant_values=np.complex128(0+0j))
    fft_image = fftshift(fft2(ifftshift(pad_image)))

    # FFT should produce one's
    assert fft_image.shape == (npix_psf, npix_psf)
    assert np.sum(fft_image) == npix_psf * npix_psf
    assert fft_image.dtype == np.complex128

    vis = degrid(fft_image[:, :, None], uvw, freqs,
                 _kernel_meta(cell_size,  npix_psf))

    assert vis.shape == (nrow, nchan, 1)

    # I^D = R+(V)
    meta = _kernel_meta(cell_size,  npix_psf)
    grid_vis = grid(vis, uvw, flags, weights, freqs, meta,
                    ny=npix_psf, nx=npix_psf)
    assert grid_vis.shape == (npix_psf, npix_psf, 1)
    assert grid_vis.dtype == np.complex128

    dirty = fftshift(ifft2(ifftshift(grid_vis[:, :, 0]))).real
    dirty = (dirty*meta.taper)[unpad, unpad]

    assert dirty.dtype == grid_vis.real.dtype

    # PSF = R+(1)
    grid_unity = grid(np.ones_like(vis), uvw, flags, weights, freqs, meta,
                      ny=npix_psf, nx=npix_psf)

    psf = fftshift(ifft2(ifftshift(grid_unity[:, :, 0]))).real
    psf = (psf*meta.taper)[unpad, unpad]

    assert psf.shape == (npix, npix)
    assert psf.dtype == grid_unity.real.dtype

    # Test that we have gridded something
    assert np.any(dirty != 0.0)
    assert np.any(psf != 0.0)

    # Normalise by PSF
    psf_max = psf.max()
    norm_psf = psf / psf_max
    norm_dirty = dirty / psf_max

    psf, dirty = norm_psf, norm_dirty

    try:
        if not plot:
            raise ValueError("Plotting disabled")

        import matplotlib.pyplot as plt
    except (ImportError, ValueError):
        pass
    else:
        plt.subplot(1, 3, 1)
        plt.imshow(dirty, cmap="cubehelix")
        plt.title("DIRTY")
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.imshow(psf, cmap="cubehelix")
        plt.title("PSF")
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.imshow(psf - dirty, cmap="cubehelix")
        plt.title("PSF - DIRTY")
        plt.colorbar()

        plt.show(True)

    assert psf.shape == dirty.shape == (npix, npix)
    # Should be very much the same
    assert_array_almost_equal(psf, dirty)


@pytest.mark.parametrize("support", [11])
@pytest.mark.parametrize("cell_size", [.1])
@pytest.mark.parametrize("npix", [257])
@pytest.mark.parametrize("wlayers", [7])
@pytest.mark.parametrize("oversampling", [1])
@pytest.mark.parametrize("lm", [(0.0, 0.0)])
def test_landman(support, cell_size, npix, wlayers,
                 oversampling, lm):
    from numpy.fft import fftshift, fft2, ifftshift, ifft2

    vis = np.array([[[1.0 + 0.0j]]], dtype=np.complex128)
    uvw = np.zeros((1, 3), dtype=np.float64)
    flag = np.zeros_like(vis, dtype=np.uint8)
    weight = np.ones_like(flag, dtype=np.float64)
    freqs = np.array([1.0], dtype=np.float64)
    maxw = 30000

    meta = wplanes(wlayers, cell_size, support, maxw,
                   npix, oversampling,
                   lm, freqs)

    grid_vis = grid(vis, uvw, flag, weight, freqs, meta,
                    ny=2*npix - 1, nx=2*npix - 1)

    # assert_array_almost_equal(grid_vis, grid_vis.T)

    try:
        import matplotlib.pyplot as plt
    except (ImportError, ValueError):
        pass
    else:
        plt.imshow(np.abs(grid_vis[:, :, 0]))
        plt.show()

        dirty = fftshift(ifft2(ifftshift(grid_vis[:, :, 0]))).real

        plt.imshow(dirty)
        plt.show()

        unpad = slice((npix // 2), -(npix // 2))
        dirty = dirty[unpad, unpad]*meta.taper

        plt.imshow(np.abs(dirty))
        plt.show()
