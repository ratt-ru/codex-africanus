# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from numba import types
import numpy as np

from africanus.util.numba import generated_jit, njit
from africanus.util.docs import DocstringTemplate


def log_si_factory(log_si_type):
    if log_si_type == "array":
        def getter_impl(log_si, src, corr):
            return log_si[src, corr]

        def reshaper_impl(log_si, shape):
            return log_si.reshape(shape)

    elif log_si_type == "bool":
        def getter_impl(log_si, src, corr):
            return log_si

        def reshaper_impl(log_si, shape):
            return log_si
    else:
        raise ValueError("log_is_type not in ('array', 'bool')")

    getter = njit(nogil=True, cache=True)(getter_impl)
    reshaper = njit(nogil=True, cache=True)(reshaper_impl)

    return getter, reshaper


def corr_getter_factory(ncorrs):
    if ncorrs == 0:
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
def spectra(stokes, spi, log_si, ref_freq, frequency):
    arg_dtypes = tuple(np.dtype(a.dtype.name) for a
                       in (stokes, spi, ref_freq, frequency))
    dtype = np.result_type(*arg_dtypes)

    ncorr = stokes.ndim - 1

    if ncorr > 2:
        raise ValueError("More than two correlation dimensions "
                         "are not supported")

    if spi.ndim != 2 + ncorr:
        raise ValueError("Invalid number of spi dims relative to stokes")

    if isinstance(log_si, types.npytypes.Array) and log_si.ndim != 1 + ncorr:
        raise ValueError("Invalid number of log_si dims relative to stokes")

    if isinstance(log_si, types.npytypes.Array):
        log_si_get_fn, log_si_reshape_fn = log_si_factory("array")
    elif isinstance(log_si, types.scalars.Boolean):
        log_si_get_fn, log_si_reshape_fn = log_si_factory("bool")
    else:
        raise ValueError("log_si must be an ndarray or scalar bool")

    corr_getter = corr_getter_factory(ncorr)

    def impl(stokes, spi, log_si, ref_freq, frequency):
        if not (stokes.shape[0] == spi.shape[0] == ref_freq.shape[0]):
            raise ValueError("first dimensions of stokes, spi, "
                             "log_si and ref_freq don't match.")

        nsrc = stokes.shape[0]
        nchan = frequency.shape[0]
        nspi = spi.shape[1]

        ncorr = corr_getter(stokes.shape[1:])

        spectral_model = np.empty((nsrc, nchan, ncorr), dtype=dtype)
        flog_si = log_si_reshape_fn(log_si, (nsrc, ncorr))
        fstokes = stokes.reshape((nsrc, ncorr))
        fspi = spi.reshape((nsrc, nspi, ncorr))

        for s in range(nsrc):
            rf = ref_freq[s]

            # Initialise with base polynomial value
            for f in range(frequency.shape[0]):
                for c in range(ncorr):
                    if log_si_get_fn(flog_si, s, c):
                        spectral_model[s, f, c] = np.log(fstokes[s, c])
                    else:
                        spectral_model[s, f, c] = fstokes[s, c]

            # Evaluate terms of the polynomial
            for f, nu in enumerate(frequency):
                for c in range(ncorr):
                    if log_si_get_fn(flog_si, s, c):
                        for si in range(nspi):
                            term = fspi[s, si, c]
                            term *= np.log(nu/rf)**(si + 1)
                            spectral_model[s, f, c] += term
                    else:
                        for si in range(nspi):
                            term = fspi[s, si, c]
                            term *= (nu/rf - 1)**(si + 1)
                            spectral_model[s, f, c] += term

            # Finalise
            for f in range(frequency.shape[0]):
                for c in range(ncorr):
                    if log_si_get_fn(flog_si, s, c):
                        spectral_model[s, f, c] = (
                                np.exp(spectral_model[s, f, c]))

        return spectral_model.reshape((nsrc, nchan) + stokes.shape[1:])

    return impl


SPECTRA_DOCS = DocstringTemplate("""
Produces a spectral model from a polynomial expansion.

Parameters
----------
stokes : :class:`numpy.ndarray`
    stokes parameters of shape :code:`(source, corr_1, corr_2)`
spi : :class:`numpy.ndarray`
    spectral index of shape :code:`(source, spi_comps, corr_1, corr_2)`
log_si : :class:`numpy.ndarray` or bool
    boolean array of shape :code:`(source, corr_1, corr_2)`
    indicating whether logarithmic (True) or ordinary (False)
    polynomials should be used.
ref_freq : :class:`numpy.ndarray`
    Source reference frequencies of shape :code:`(source,)`
frequency : :class:`numpy.ndarray`
    frequencies of shape :code:`(chan,)`

Notes
-----
Between zero and two correlation dimensions are supported,
but the number of correlations in `stokes`, `spi` and `log_si`
must agree.

Returns
-------
spectral_model : :class:`numpy.ndarray`
    Spectral Model of shape :code:`(source, chan, corr_1, corr_2)`
""")

try:
    spectra.__doc__ = SPECTRA_DOCS.substitute(
                            array_type=":class:`numpy.ndarray")
except AttributeError:
    pass
