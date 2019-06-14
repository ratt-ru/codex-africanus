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
@pytest.mark.parametrize("lm_shift", [(1e-8, 1e-8)])
def test_degridder(support, spheroidal_support, npix,
                   wlayers, maxw, cell_size,
                   oversampling, lm_shift):

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
                   lm_shift, freqs)

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
@pytest.mark.parametrize("lm_shift", [(1e-8, 1e-8)])
def test_gridder(support, spheroidal_support, npix,
                 wlayers, maxw, cell_size,
                 oversampling, lm_shift):

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
                   lm_shift, freqs)

    grid_ = grid(vis, uvw, flags, weights, freqs, meta, ny=npix, nx=npix)

    assert grid_.shape == (npix, npix, ncorr)


@pytest.mark.xfail(reason="Dirty vs PSF scaling is off for some reason")
@pytest.mark.parametrize("support", [11])
@pytest.mark.parametrize("spheroidal_support", [111])
@pytest.mark.parametrize("npix", [513])
@pytest.mark.parametrize("wlayers", [7])
@pytest.mark.parametrize("maxw", [30000])
@pytest.mark.parametrize("oversampling", [11])
@pytest.mark.parametrize("lm_shift", [(1e-8, 1e-8)])
@pytest.mark.parametrize("plot", [True])
def test_psf_subtraction(support, spheroidal_support,
                         npix, wlayers, maxw,
                         oversampling, lm_shift,
                         plot):

    def _kernel_meta(cell_size, npix):
        return wplanes(wlayers, cell_size, support, maxw,
                       npix, oversampling,
                       lm_shift, freqs)

    from numpy.fft import fftshift, fft2, ifftshift, ifft2

    nrow = 10
    nchan = 8
    ncorr = 1

    freqs = np.linspace(.856e9, 2*.856e9, nchan)
    wavelengths = lightspeed / freqs

    # Random UVW coordinates
    uvw = np.empty((nrow, 3), dtype=np.float64)
    uvw[:, :2] = (rf((nrow, 2)) - 0.5)*10000
    uvw[:, 2] = (rf((nrow,)) - 0.5)*10

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

    # V = R(I)
    # Created a padded image, FFT into the centre
    padding = ((npad_psf, npad_psf), (npad_psf, npad_psf))
    fft_image = np.pad(fftshift(fft2(ifftshift(image))), padding,
                       mode='constant', constant_values=np.complex128(0+0j))

    # FFT should produce one's over the central npix x npix
    assert fft_image.shape == (npix_psf, npix_psf)
    assert np.sum(fft_image) == npix*npix
    assert fft_image.dtype == np.complex128

    ref_wave = wavelengths[wavelengths.shape[0] // 2]

    vis = degrid(fft_image[:, :, None], uvw, freqs,
                 _kernel_meta(2*cell_size,  npix_psf))

    assert vis.shape == (nrow, nchan, 1)

    # I^D = R+(V)
    grid_vis = grid(vis, uvw, flags, weights, freqs,
                    _kernel_meta(2*cell_size,  npix),
                    ny=npix, nx=npix)
    assert grid_vis.shape == (npix, npix, 1)
    assert grid_vis.dtype == np.complex128

    padding = ((npad_psf, npad_psf), (npad_psf, npad_psf), (0, 0))
    grid_vis = np.pad(grid_vis, padding, mode='constant',
                      constant_values=grid_vis.dtype.type(0))

    assert grid_vis.shape == (npix_psf, npix_psf, 1)
    assert grid_vis.dtype == vis.dtype

    dirty = fftshift(ifft2(ifftshift(grid_vis[:, :, 0]))).real

    assert dirty.dtype == grid_vis.real.dtype

    # PSF = R+(1)
    grid_unity = grid(np.ones_like(vis), uvw, flags, weights, freqs,
                      _kernel_meta(cell_size,  npix_psf),
                      ny=npix_psf, nx=npix_psf)

    psf = fftshift(ifft2(ifftshift(grid_unity[:, :, 0]))).real

    assert psf.shape == (npix_psf, npix_psf)
    assert psf.dtype == grid_unity.real.dtype

    # Test that we have gridded something
    assert np.any(dirty != 0.0)
    assert np.any(psf != 0.0)

    norm_psf = psf / psf.max()
    norm_dirty = dirty / psf.max()

    psf, dirty = norm_psf, norm_dirty

    # Extract the centre of the PSF and the dirty image
    ex = slice(npix - npad_psf, npix + npad_psf + 1)
    centre_psf = psf[ex, ex].copy()
    centre_dirty = dirty[ex, ex].copy()

    try:
        if not plot:
            raise ValueError("Plotting disabled")

        import matplotlib.pyplot as plt
    except (ImportError, ValueError):
        pass
    else:
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

    assert centre_psf.shape == centre_dirty.shape == (npix, npix)
    # Should be very much the same
    assert_array_almost_equal(centre_psf, centre_dirty)
