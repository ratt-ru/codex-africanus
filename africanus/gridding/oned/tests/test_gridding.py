import numba
import numpy as np
import pytest

from africanus.constants import c as lightspeed
from africanus.constants import minus_two_pi_over_c

from africanus.gridding.oned.gridding import grid
from africanus.filters.kaiser_bessel_filter import (
                    kaiser_bessel,
                    kaiser_bessel_fourier,
                    kaiser_bessel_with_sinc)

from scipy.fftpack import ifft, fftshift, ifftshift


def kaiser_bessel_taper(full_support, oversample, beta, nx):
        kb = kaiser_bessel_with_sinc(full_support, oversample, beta)
        kbshift = np.fft.fftshift(kb)

        width = nx * oversample
        # assert width % 2 == 1
        nx_odd = nx % 2 == 1

        # Put the first and last halves of the shifted Kaiser Bessel
        # at each end of the output buffer, then FFT
        buf = np.zeros(width, dtype=kb.dtype)
        buf[:kbshift.size // 2] = kbshift[:kbshift.size // 2]
        buf[-kbshift.size // 2:] = kbshift[-kbshift.size // 2:]

        x = np.fft.ifft(buf).real

        # First half of the taper
        half = x[:nx_odd + (nx // 2)]

        print("Full", nx, "Half", half.size)

        taper = np.empty(nx, dtype=kb.dtype)

        if nx_odd:
            taper[:1 + (nx // 2)] = half[::-1]
            taper[nx // 2:] = half
            assert np.allclose(taper[:nx // 2], taper[1 + (nx // 2):][::-1])
        else:
            taper[:nx // 2] = half[::-1]
            taper[nx // 2:] = half
            assert np.allclose(taper[:nx // 2], taper[nx // 2:])


        return taper*oversample


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
    _ARCSEC2RAD = np.deg2rad(1.0/(60*60))
    filter_width = 7
    oversample = 15
    beta = 2.34*filter_width
    cell_size = 8.
    cell_size_rad = _ARCSEC2RAD*cell_size
    nx = 65

    vis = np.asarray([[1.0 + 2.0j]], dtype=np.complex128)
    uvw = np.asarray([[10000.0]], dtype=np.float64)
    freq = np.asarray([.856e9], dtype=np.float64)
    ref_wave = freq / lightspeed

    conv_filter = kaiser_bessel_with_sinc(filter_width, oversample, beta)
    taper = kaiser_bessel_taper(filter_width, oversample, beta, nx)
    oned_grid = np.zeros(nx, dtype=np.complex128)

    grid(vis, uvw, ref_wave, conv_filter, oversample, cell_size, oned_grid)

    # FFT
    oned_grid_fft = fftshift(ifft(ifftshift(oned_grid)))
    # Normalies by FFT factor
    oned_grid_fft *= nx
    # Apply the taper
    # oned_grid_fft /= taper

    # Do the DFT
    lm = np.arange(nx, dtype=np.float64) - (nx // 2)
    lm *= cell_size_rad
    print("lm", lm)
    # Centre point must be 0.0
    assert lm[nx // 2] == 0.0

    dft_grid = np.zeros((oned_grid.shape[0], freq.shape[0]),
                        dtype=np.complex64)

    vis_to_im_impl(vis, uvw, lm, freq, dft_grid)

    phase = 2.*np.pi*(lm*uvw[0][0])*freq[0]/lightspeed
    np_vis = (np.cos(phase) * vis[0, 0].real -
              np.sin(phase) * vis[0, 0].imag)

    assert np.allclose(np_vis.real, dft_grid.squeeze().real)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pass
    else:
        plt.figure()

        arrays = [dft_grid.squeeze().real, oned_grid_fft.real, taper]
        names = ["dft", "fft", "taper"]

        for name, array in zip(names, arrays):
            plt.plot(array)

        plt.legend(names)
        plt.show()
