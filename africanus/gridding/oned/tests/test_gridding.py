import numba
import numpy as np
import pytest

from africanus.constants import c as lightspeed
from africanus.constants import minus_two_pi_over_c


def vis_to_im_impl(vis, uvw, lm, frequency, im_of_vis):
    # For each source
    for source in range(lm.shape[0]):
        l = lm[source]
        # n = np.sqrt(1.0 - l ** 2 - m ** 2) - 1.0
        # For each uvw coordinate
        for row in range(uvw.shape[0]):
            u = uvw[row]

            # e^(-2*pi*(l*u + m*v + n*w)/c)
            real_phase = -minus_two_pi_over_c * (l * u)

            # Multiple in frequency for each channel
            for chan in range(frequency.shape[0]):
                p = real_phase * frequency[chan]

                im_of_vis[source,
                          chan] += (np.cos(p) * vis[row, chan].real -
                                    np.sin(p) * vis[row, chan].imag)
                # Note for the adjoint we don't need the imaginary part
                # and we can elide the call to exp

    return im_of_vis


def test_oned_gridding():
    from africanus.gridding.oned.gridding import grid
    from africanus.filters.kaiser_bessel_filter import (
                        kaiser_bessel,
                        kaiser_bessel_fourier)

    _ARCSEC2RAD = np.deg2rad(1.0/(60*60))
    filter_width = 7
    oversample = 1
    beta = 2.34
    cell_size = 8.
    grid_size = 21
    cell_size_rad = _ARCSEC2RAD*cell_size

    vis = np.asarray([[1.0 + 2.0j]], dtype=np.complex128)
    uvw = np.asarray([[10000.0]], dtype=np.float64)
    freq = np.asarray([.856e9], dtype=np.float64)
    ref_wave = freq / lightspeed
    width = filter_width*oversample

    u = np.arange(width, dtype=np.float64) - width // 2

    conv_filter = kaiser_bessel(u, width, beta)
    oned_grid = np.empty(grid_size, dtype=np.complex128)

    grid(vis, uvw, ref_wave, conv_filter, oversample, cell_size, oned_grid)

    lm_extents = cell_size_rad*(grid_size // 2)
    lm = np.linspace(-lm_extents, lm_extents, grid_size)
    dft_grid = np.zeros((oned_grid.shape[0], freq.shape[0]),
                        dtype=oned_grid.dtype)

    vis_to_im_impl(vis, uvw, lm, freq, dft_grid)
    oned_grid_fft = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(oned_grid)))
    oned_grid_fft *= grid_size

    u = (np.arange(grid_size, dtype=np.float64) - (grid_size // 2)) / grid_size
    taper = kaiser_bessel_fourier(u, filter_width, beta*filter_width)
    oned_grid_fft /= taper

    print(np.stack([dft_grid.squeeze(), oned_grid_fft]))
    print(dft_grid.squeeze().real - oned_grid_fft.real)

