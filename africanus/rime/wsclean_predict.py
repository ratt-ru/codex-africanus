# -*- coding: utf-8 -*-

import numpy as np

from africanus.constants import two_pi_over_c
from africanus.util.numba import generated_jit
from africanus.model.wsclean.spec_model import spectra


@generated_jit(nopython=True, nogil=True, cache=True)
def wsclean_predict(uvw, lm, flux, coeffs, log_poly, ref_freq, frequency):
    arg_dtypes = tuple(np.dtype(a.dtype.name) for a
                       in (uvw, lm, flux, coeffs, ref_freq, frequency))
    dtype = np.result_type(np.complex64, *arg_dtypes)

    n1 = lm.dtype(1)

    def impl(uvw, lm, flux, coeffs, log_poly, ref_freq, frequency):
        nrow = uvw.shape[0]
        nchan = frequency.shape[0]
        ncorr = 1

        spectrum = spectra(flux, coeffs, log_poly, ref_freq, frequency)
        nsrc = spectrum.shape[0]

        vis = np.zeros((nrow, nchan, ncorr), dtype=dtype)

        for s in range(nsrc):
            l = lm[s, 0]  # noqa
            m = lm[s, 1]
            n = np.sqrt(n1 - l*l - m*m) - n1

            for r in range(nrow):
                u = uvw[r, 0]
                v = uvw[r, 1]
                w = uvw[r, 2]

                real_phase = two_pi_over_c*(u*l + v*m + w*n)

                for f in range(nchan):
                    p = real_phase * frequency[f]
                    re = np.cos(p) * spectrum[s, f]
                    im = np.sin(p) * spectrum[s, f]

                    vis[r, f, 0] += re + im*1j

        return vis

    return impl
