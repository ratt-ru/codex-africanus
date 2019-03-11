# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from africanus.util.numba import generated_jit, njit
from africanus.util.docs import DocstringTemplate


def numpy_spectral_model(stokes, spi, ref_freq, frequency):
    pass


def corr_getter_factory(ncorrdims):
    if ncorrdims == 0:
        def impl(corr_shape):
            return 1
    else:
        def impl(corr_shape):
            ncorrs = 1

            for c in corr_shape:
                ncorrs *= c

            return ncorrs

    return njit(nogil=True, cache=True)(impl)


@generated_jit(nopython=True, nogil=True, cache=True)
def spectral_model(stokes, spi, ref_freq, frequency):
    arg_dtypes = tuple(np.dtype(a.dtype.name) for a
                       in (stokes, spi, ref_freq, frequency))
    dtype = np.result_type(*arg_dtypes)

    ncorrdims = stokes.ndim - 1
    corr_get_fn = corr_getter_factory(ncorrdims)

    if spi.ndim == 1:
        def impl(stokes, spi, ref_freq, frequency):
            nsrc = stokes.shape[0]
            nchan = frequency.shape[0]
            ncorrs = corr_get_fn(stokes.shape[1:])

            spectral_model = np.empty((nsrc, nchan), dtype=dtype)

            for s in range(nsrc):
                rf = ref_freq[s]
                src_spi = spi[s]
                flux = stokes[s]

                for f in range(nchan):
                    spectral_model[s, f] = flux*(frequency[f]/rf)**src_spi

    elif spi.ndim == 1:
        def impl(stokes, spi, ref_freq, frequency):
            nsrc = stokes.shape[0]
            nchan = frequency.shape[0]
            ncorrs = corr_get_fn(stokes.shape[1:])
            nspi = spi.shape[1]

            spectral_model = np.empty((nsrc, nchan), dtype=dtype)

            for s in range(nsrc):
                rf = ref_freq[s]
                flux = stokes[s]

                for f in range(nchan):
                    log_freq = np.log(frequency[f])

                    for si in range(nspi):
                        spectral_model[s, f] += spi[s, si]*log_freq**si

                    spectral_model[s, f] = np.exp(spectral_model[s, f])

            return spectral_model
    else:
        raise ValueError("spi.ndim not in (1, 2)")

    return impl
