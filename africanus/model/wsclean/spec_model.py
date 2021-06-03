# -*- coding: utf-8 -*-
from numba import types
import numpy as np

from africanus.util.numba import generated_jit
from africanus.util.docs import DocstringTemplate


def ordinary_spectral_model(I, coeffs, log_poly, ref_freq, freq):  # noqa: E741
    """ Numpy ordinary polynomial implementation """
    coeffs_idx = np.arange(1, coeffs.shape[1] + 1)
    # (source, chan, coeffs-comp)
    term = (freq[None, :, None] / ref_freq[:, None, None]) - 1.0
    term = term**coeffs_idx[None, None, :]
    term = coeffs[:, None, :]*term
    return I[:, None] + term.sum(axis=2)


def log_spectral_model(I, coeffs, log_poly, ref_freq, freq):  # noqa: E741
    """ Numpy logarithmic polynomial implementation """
    # No negative flux
    I = np.where(log_poly == False, 1.0, I)  # noqa: E741, E712
    coeffs_idx = np.arange(1, coeffs.shape[1] + 1)
    # (source, chan, coeffs-comp)
    term = np.log(freq[None, :, None] / ref_freq[:, None, None])
    term = term**coeffs_idx[None, None, :]
    term = coeffs[:, None, :]*term
    return np.exp(np.log(I)[:, None] + term.sum(axis=2))


@generated_jit(nopython=True, nogil=True, cache=True)
def _check_log_poly_shape(coeffs, log_poly):
    if isinstance(log_poly, types.npytypes.Array):
        def impl(coeffs, log_poly):
            if coeffs.shape[0] != log_poly.shape[0]:
                raise ValueError("coeffs.shape[0] != log_poly.shape[0]")
    elif isinstance(log_poly, types.scalars.Boolean):
        def impl(coeffs, log_poly):
            pass
    else:
        raise ValueError("log_poly must be ndarray or bool")

    return impl


@generated_jit(nopython=True, nogil=True, cache=True)
def _log_polynomial(log_poly, s):
    if isinstance(log_poly, types.npytypes.Array):
        def impl(log_poly, s):
            return log_poly[s]
    elif isinstance(log_poly, types.scalars.Boolean):
        def impl(log_poly, s):
            return log_poly
    else:
        raise ValueError("log_poly must be ndarray or bool")

    return impl


@generated_jit(nopython=True, nogil=True, cache=True)
def spectra(I, coeffs, log_poly, ref_freq, frequency):  # noqa: E741
    arg_dtypes = tuple(np.dtype(a.dtype.name) for a
                       in (I, coeffs, ref_freq, frequency))
    dtype = np.result_type(*arg_dtypes)

    def impl(I, coeffs, log_poly, ref_freq, frequency):  # noqa: E741
        if not (I.shape[0] == coeffs.shape[0] == ref_freq.shape[0]):
            print(I.shape, coeffs.shape, ref_freq.shape)
            raise ValueError("first dimensions of I, coeffs "
                             "and ref_freq don't match.")

        _check_log_poly_shape(coeffs, log_poly)

        nsrc = I.shape[0]
        nchan = frequency.shape[0]
        ncoeffs = coeffs.shape[1]

        spectral_model = np.empty((nsrc, nchan), dtype=dtype)

        for s in range(nsrc):
            rf = ref_freq[s]

            if _log_polynomial(log_poly, s):
                for f in range(frequency.shape[0]):
                    nu = frequency[f]

                    flux = I[s]

                    if flux <= 0.0:
                        raise ValueError("log polynomial flux must be > 0")

                    # Initialise with base polynomial value
                    spectral_model[s, f] = np.log(flux)

                    for c in range(ncoeffs):
                        term = coeffs[s, c] * np.log(nu/rf)**(c + 1)
                        spectral_model[s, f] += term

                    spectral_model[s, f] = np.exp(spectral_model[s, f])
            else:
                for f in range(frequency.shape[0]):
                    nu = frequency[f]

                    # Initialise with base polynomial value
                    spectral_model[s, f] = I[s]

                    for c in range(ncoeffs):
                        term = coeffs[s, c]
                        term *= ((nu/rf) - 1.0)**(c + 1)
                        spectral_model[s, f] += term

        return spectral_model

    return impl


SPECTRA_DOCS = DocstringTemplate(r"""
Produces a spectral model from a polynomial expansion of
a wsclean file model. Depending on how `log_poly` is set
ordinary or logarithmic polynomials are used to produce
the expansion:

.. math::

    & flux(\lambda) =
      I_{0} + \sum\limits_{c=0} \textrm{coeffs}(c)
              ({\lambda/\lambda_{ref}} - 1)^{c+1}
              \\
    & flux(\lambda) =
      \exp \left( \log I_{0} +
              \sum\limits_{c=0} \textrm{coeffs}(c)
              \log({\lambda/\lambda_{ref}})^{c+1}
            \right) \\


See the `WSClean Component List
<https://sourceforge.net/p/wsclean/wiki/ComponentList/>`_
for further details.

Parameters
----------
I : $(array_type)
    flux density in Janskys at the reference frequency
    of shape :code:`(source,)`
coeffs : $(array_type)
    Polynomial coefficients for each source of
    shape :code:`(source, comp)`
log_poly : $(array_type) or bool
    boolean array of shape :code:`(source, )`
    indicating whether logarithmic (True) or ordinary (False)
    polynomials should be used.
ref_freq : $(array_type)
    Source reference frequencies of shape :code:`(source,)`
frequency : $(array_type)
    frequencies of shape :code:`(chan,)`

See Also
--------
africanus.model.wsclean.load

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
