# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from africanus.util.numba import generated_jit, njit
from africanus.util.docs import DocstringTemplate


def numpy_spectral_model(stokes, spi, ref_freq, frequency):
    spi_exps = np.arange(1, spi.shape[1] + 1)
    freq_ratio = (frequency[None, :] / ref_freq[:, None]) - 1.0
    term = spi[:, :, None] * freq_ratio[:, None, :]**(spi_exps[None, :, None])
    return stokes[:, None] + term.sum(axis=1)


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
def spectral_model(stokes, spi, ref_freq, frequency, base=None):
    arg_dtypes = tuple(np.dtype(a.dtype.name) for a
                       in (stokes, spi, ref_freq, frequency))
    dtype = np.result_type(*arg_dtypes)

    ncorrdims = stokes.ndim - 1
    corr_get_fn = corr_getter_factory(ncorrdims)  # noqa

    def impl(stokes, spi, ref_freq, frequency, base=None):
        nsrc = stokes.shape[0]
        nchan = frequency.shape[0]
        nspi = spi.shape[1]

        spectral_model = np.empty((nsrc, nchan), dtype=dtype)

        for s in range(nsrc):
            flux = stokes[s]
            rf = ref_freq[s]

            for f in range(nchan):
                spectral_model[s, f] = flux
                freq_ratio = (frequency[f] / rf) - 1.0

                for si in range(0, nspi):
                    spectral_model[s, f] += spi[s, si] * freq_ratio**(si + 1)

        return spectral_model

    return impl


SPECTRAL_MODEL_DOC = DocstringTemplate(r"""
Calculate the spectral model.

Parameters
----------
stokes : $(array_type)
    Stokes parameters of shape :code:`(source,)`
spi : $(array_type)
    Spectral index of shape :code:`(source, spi-comps)`
ref_freq : $(array_type)
    Reference frequencies of shape :code:`(source,)`
frequencies : $(array_type)
    Frequencies of shape :code:`(chan,)`
base : {"std", "log"}

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
