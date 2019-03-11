# -*- coding: utf-8 -*-
from numba import types
import numpy as np

from africanus.compatibility import string_types
from africanus.util.numba import generated_jit, njit
from africanus.util.docs import DocstringTemplate


def ordinary_spectral_model(I, spi, log_si, freq, ref_freq):
    """ Numpy ordinary polynomial implementation """
    spi_idx = np.arange(1, spi.shape[1] + 1)
    # (source, chan, spi-comp)
    term = (freq[None, :, None] / ref_freq[:, None, None]) - 1.0
    term = term**spi_idx[None, None, :]
    term = spi[:, None, :]*term
    return I[:, None] + term.sum(axis=2)


def log_spectral_model(I, spi, log_si, freq, ref_freq):
    """ Numpy logarithmic polynomial implementation """
    # No negative flux
    I = np.where(log_si == False, 1.0, I)  # noqa
    spi_idx = np.arange(1, spi.shape[1] + 1)
    # (source, chan, spi-comp)
    term = np.log(freq[None, :, None] / ref_freq[:, None, None])
    term = term**spi_idx[None, None, :]
    term = spi[:, None, :]*term
    return np.exp(np.log(I)[:, None] + term.sum(axis=2))


@generated_jit(nopython=True, nogil=True, cache=True)
def _check_log_si_shape(spi, log_si):
    if isinstance(log_si, types.npytypes.Array):
        def impl(spi, log_si):
            if spi.shape[0] != log_si.shape[0]:
                raise ValueError("spi.shape[0] != log_si.shape[0]")
    elif isinstance(log_si, types.scalars.Boolean):
        def impl(spi, log_si):
            pass
    else:
        raise ValueError("log_si must be ndarray or bool")

    return impl


@generated_jit(nopython=True, nogil=True, cache=True)
def _log_polynomial(log_si, s):
    if isinstance(log_si, types.npytypes.Array):
        def impl(log_si, s):
            return log_si[s]
    elif isinstance(log_si, types.scalars.Boolean):
        def impl(log_si, s):
            return log_si
    else:
        raise ValueError("log_si must be ndarray or bool")

    return impl


@generated_jit(nopython=True, nogil=True, cache=True)
def spectra(I, spi, log_si, ref_freq, frequency):
    arg_dtypes = tuple(np.dtype(a.dtype.name) for a
                       in (I, spi, ref_freq, frequency))
    dtype = np.result_type(*arg_dtypes)

    def impl(I, spi, log_si, ref_freq, frequency):
        if not (I.shape[0] == spi.shape[0] == ref_freq.shape[0]):
            raise ValueError("first dimensions of I, spi "
                             "and ref_freq don't match.")

        _check_log_si_shape(spi, log_si)

        nsrc = I.shape[0]
        nchan = frequency.shape[0]
        nspi = spi.shape[1]

        spectral_model = np.empty((nsrc, nchan), dtype=dtype)

        for s in range(nsrc):
            rf = ref_freq[s]

            if _log_polynomial(log_si, s):
                for f in range(frequency.shape[0]):
                    # Initialise with base polynomial value
                    spectral_model[s, f] = np.log(I[s])

                    for si in range(nspi):
                        term = spi[s, si]
                        term *= np.log(frequency[f]/rf)**(si + 1)
                        spectral_model[s, f] += term

                    spectral_model[s, f] = np.exp(spectral_model[s, f])
            else:
                for f in range(frequency.shape[0]):
                    nu = frequency[f]

                    # Initialise with base polynomial value
                    spectral_model[s, f] = I[s]

                    for si in range(nspi):
                        term = spi[s, si]
                        term *= ((frequency[f]/rf) - 1.0)**(si + 1)
                        spectral_model[s, f] += term

        return spectral_model

    return impl


SPECTRA_DOCS = DocstringTemplate(r"""
Produces a spectral model from a polynomial expansion of
a wsclean file model. Depending on how `log_si` is set
ordinary or logarithmic polynomials are used to produce
the expansion:

.. math::

    & flux(\lambda) =
      \textrm{stokes} +
              \sum\limits_{si=0} \textrm{spi}(si)
              ({\lambda/\lambda_{ref}} - 1)^{si+1}
              \\
    & flux(\lambda) =
      \exp \left( \log \textrm{stokes} +
              \sum\limits_{si=0} \textrm{spi}(si)
              \log({\lambda/\lambda_{ref}})^{si+1}
            \right) \\


Parameters
----------
I : $(array_type)
    I of shape :code:`(source,)`
spi : $(array_type)
    spectral index for each source of shape :code:`(source, spi)`
log_si : $(array_type) or bool
    boolean array of shape :code:`(source, )`
    indicating whether logarithmic (True) or ordinary (False)
    polynomials should be used.
ref_freq : $(array_type)
    Source reference frequencies of shape :code:`(source,)`
frequency : $(array_type)
    frequencies of shape :code:`(chan,)`

Returns
-------
spectral_model : $(array_type)
    Spectral Model of shape :code:`(source, chan)`
""")

try:
    spectra.__doc__ = SPECTRA_DOCS.substitute(
                            array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass

