# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from africanus.util.numba import generated_jit, njit
from africanus.util.docs import DocstringTemplate


def numpy_spectral_model(stokes, spi, ref_freq, frequency):
    if spi.ndim == 1:
        freq_ratio = frequency[None, :] / ref_freq[:, None]
        return stokes[:, None] * freq_ratio**spi[:, None]
    elif spi.ndim == 2:
        spi_exps = np.arange(spi.shape[1])[None, :, None]
        term = np.log10(frequency / 1e9)[None, None, :] ** spi_exps
        term = spi[:, :, None] * term
        return stokes[:, None] * 10**term.sum(axis=1)
    else:
        raise ValueError("spi.ndim not in (1, 2)")


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

    ONE_GHZ = arg_dtypes[-1].type(1e9)

    ncorrdims = stokes.ndim - 1
    corr_get_fn = corr_getter_factory(ncorrdims)  # noqa

    if spi.ndim == 1:
        def impl(stokes, spi, ref_freq, frequency):
            nsrc = stokes.shape[0]
            nchan = frequency.shape[0]

            spectral_model = np.empty((nsrc, nchan), dtype=dtype)

            for s in range(nsrc):
                rf = ref_freq[s]
                src_spi = spi[s]
                flux = stokes[s]

                for f in range(nchan):
                    spectral_model[s, f] = flux*(frequency[f]/rf)**src_spi

            return spectral_model

    elif spi.ndim == 2:
        def impl(stokes, spi, ref_freq, frequency):
            nsrc = stokes.shape[0]
            nchan = frequency.shape[0]
            nspi = spi.shape[1]

            spectral_model = np.empty((nsrc, nchan), dtype=dtype)

            for s in range(nsrc):
                flux = stokes[s]

                for f in range(nchan):
                    spectral_model[s, f] = spi[s, 0]
                    log_freq = np.log10(frequency[f] / ONE_GHZ)

                    for si in range(1, nspi):
                        spectral_model[s, f] += spi[s, si] * log_freq**si

                    spectral_model[s, f] = flux * 10**spectral_model[s, f]

            return spectral_model
    else:
        raise ValueError("spi.ndim not in (1, 2)")

    return impl


SPECTRAL_MODEL_DOC = DocstringTemplate(r"""
Calculate the spectral model.

Parameters
----------
stokes : $(array_type)
    Stokes parameters of shape :code:`(source,)`
spi : $(array_type)
    Spectral index of shape :code:`(source,)` or
    :code:`(source, spi-comps)`
ref_freq : $(array_type)
    Reference frequencies of shape :code:`(source,)`
frequencies : $(array_type)
    Frequencies of shape :code:`(chan,)`

Returns
-------
spectral_model : $(array_type)
    Spectral Model of shape :code:`(source, chan)`
""")

try:
    spectral_model.__doc__ = SPECTRAL_MODEL_DOC.subsitute(
                                array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass
