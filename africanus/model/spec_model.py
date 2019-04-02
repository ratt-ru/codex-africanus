# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from numba import types
import numpy as np

from africanus.compatibility import PY2, PY3
from africanus.util.numba import generated_jit, njit
from africanus.util.docs import DocstringTemplate


def numpy_spectral_model(stokes, spi, ref_freq, frequency, base):
    spi_exps = np.arange(1, spi.shape[1] + 1)
    if base in ("std", 0):
        freq_ratio = (frequency[None, :] / ref_freq[:, None]) - 1.0
        term = freq_ratio[:, None, :, None]**(spi_exps[None, :, None, None])
        term = spi[:, :, None, :] * term
        return stokes[:, None, :] + term.sum(axis=1)
    elif base in ("log", 1):
        freq_ratio = np.log(frequency[None, :] / ref_freq[:, None])
        term = freq_ratio[:, None, :, None]**(spi_exps[None, :, None, None])
        term = spi[:, :, None, :] * term
        return np.exp(np.log(stokes[:, None, :]) + term.sum(axis=1))
    elif base in ("log10", 2):
        freq_ratio = np.log10(frequency[None, :] / ref_freq[:, None])
        term = freq_ratio[:, None, :, None]**(spi_exps[None, :, None, None])
        term = spi[:, :, None, :] * term
        return 10**(np.log10(stokes[:, None]) + term.sum(axis=1))
    else:
        raise ValueError("Invalid base %s" % base)


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
def spectral_model(stokes, spi, ref_freq, frequency, base=0):
    arg_dtypes = tuple(np.dtype(a.dtype.name) for a
                       in (stokes, spi, ref_freq, frequency))
    dtype = np.result_type(*arg_dtypes)

    # numba doesn't support strings yet, support base type
    # through integers
    if isinstance(base, types.scalars.Integer):
        def is_std(base):
            return base == 0

        def is_log(base):
            return base == 1

        def is_log10(base):
            return base == 2

    elif PY3 and isinstance(base, types.misc.UnicodeType):
        def is_std(base):
            return base == "std"

        def is_log(base):
            return base == "log"

        def is_log10(base):
            return base == "log10"

    elif PY2 and isinstance(base, types.common.Opaque) and base.name == "str":
        # python 2 string support can be added when
        # https://github.com/numba/numba/issues/3323
        # is complete
        raise TypeError("String 'base' unsupported in python 2. "
                        "Use integers to specify the type of "
                        "polynomial base.")
    else:
        raise TypeError("base '%s' should be a string or integer" % base)

    is_std = njit(nogil=True, cache=True)(is_std)
    is_log = njit(nogil=True, cache=True)(is_log)
    is_log10 = njit(nogil=True, cache=True)(is_log10)

    ncorrdims = stokes.ndim - 1
    corr_get_fn = corr_getter_factory(ncorrdims)

    def impl(stokes, spi, ref_freq, frequency, base=0):
        nsrc = stokes.shape[0]
        nchan = frequency.shape[0]
        nspi = spi.shape[1]
        ncorr = corr_get_fn(stokes.shape[1:])

        if ncorr != corr_get_fn(spi.shape[2:]):
            raise ValueError("Correlations on stokes + spi do not agree")

        spectral_model = np.empty((nsrc, nchan, ncorr), dtype=dtype)

        if is_std(base):
            for s in range(nsrc):
                rf = ref_freq[s]

                for f in range(nchan):
                    freq_ratio = (frequency[f] / rf) - 1.0

                    for c in range(0, ncorr):
                        spectral_model[s, f, c] = stokes[s, c]

                        for si in range(0, nspi):
                            term = spi[s, si, c] * freq_ratio**(si + 1)
                            spectral_model[s, f, c] += term

        elif is_log(base):
            for s in range(nsrc):
                rf = ref_freq[s]

                for f in range(nchan):
                    freq_ratio = np.log(frequency[f] / rf)

                    for c in range(0, ncorr):
                        spectral_model[s, f, c] = np.log(stokes[s, c])

                        for si in range(0, nspi):
                            term = spi[s, si, c] * freq_ratio**(si + 1)
                            spectral_model[s, f, c] += term

                        exp_spectra = np.exp(spectral_model[s, f, c])
                        spectral_model[s, f, c] = exp_spectra

        elif is_log10(base):
            for s in range(nsrc):
                rf = ref_freq[s]

                for f in range(nchan):
                    freq_ratio = np.log10(frequency[f] / rf)

                    for c in range(0, ncorr):
                        spectral_model[s, f, c] = np.log10(stokes[s, c])

                        for si in range(0, nspi):
                            term = spi[s, si, c] * freq_ratio**(si + 1)
                            spectral_model[s, f, c] += term

                        exp_spectra = 10**spectral_model[s, f, c]
                        spectral_model[s, f, c] = exp_spectra

        else:
            raise ValueError("Invalid base")

        return spectral_model

    return impl


SPECTRAL_MODEL_DOC = DocstringTemplate(r"""
Calculate the spectral model.

Parameters
----------
stokes : $(array_type)
    Stokes parameters of shape :code:`(source, corr)`
spi : $(array_type)
    Spectral index of shape :code:`(source, spi-comps, corr)`
ref_freq : $(array_type)
    Reference frequencies of shape :code:`(source,)`
frequencies : $(array_type)
    Frequencies of shape :code:`(chan,)`
base : {"std", "log", "log10"} or {0, 1, 2}.
    string or enumeration specifying the polynomial base. Defaults to 0.

    string specification of the base is only supported in python 3.
    while the corresponding integer enumerations are supported
    on all python versions.

Returns
-------
spectral_model : $(array_type)
    Spectral Model of shape :code:`(source, chan, corr)`
""")

try:
    spectral_model.__doc__ = SPECTRAL_MODEL_DOC.subsitute(
                                array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass
