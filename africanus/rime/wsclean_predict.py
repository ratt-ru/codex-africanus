# -*- coding: utf-8 -*-

import numpy as np

from africanus.constants import two_pi_over_c, c as lightspeed
from africanus.util.docs import DocstringTemplate
from africanus.util.numba import generated_jit, jit
from africanus.model.wsclean.spec_model import spectra


@jit(nopython=True, nogil=True, cache=True)
def wsclean_predict_impl(uvw, lm, source_type, gauss_shape,
                         frequency, spectrum, dtype):

    fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
    fwhminv = 1.0 / fwhm
    gauss_scale = fwhminv * np.sqrt(2.0) * np.pi / lightspeed

    nrow = uvw.shape[0]
    nchan = frequency.shape[0]
    ncorr = 1

    nsrc = spectrum.shape[0]
    n1 = lm.dtype.type(1)

    scaled_freq = frequency * frequency.dtype.type(gauss_scale)

    vis = np.zeros((nrow, nchan, ncorr), dtype=dtype)

    for s in range(nsrc):
        l = lm[s, 0]  # noqa
        m = lm[s, 1]
        n = np.sqrt(n1 - l*l - m*m) - n1

        if source_type[s] == "POINT":
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
        elif source_type[s] == "GAUSSIAN":
            emaj, emin, angle = gauss_shape[s]

            # Convert to l-projection, m-projection, ratio
            el = emaj * np.sin(angle)
            em = emaj * np.cos(angle)
            er = emin / (1.0 if emaj == 0.0 else emaj)

            for r in range(nrow):
                u = uvw[r, 0]
                v = uvw[r, 1]
                w = uvw[r, 2]

                # Compute phase term
                real_phase = two_pi_over_c*(u*l + v*m + w*n)

                # Gaussian shape term bits
                u1 = (u*em - v*el)*er
                v1 = u*el + v*em

                for f in range(nchan):
                    p = real_phase * frequency[f]
                    re = np.cos(p) * spectrum[s, f]
                    im = np.sin(p) * spectrum[s, f]

                    # Calculate gaussian shape component and multiply in
                    fu1 = u1 * scaled_freq[f]
                    fv1 = v1 * scaled_freq[f]
                    shape = np.exp(-(fu1 * fu1 + fv1 * fv1))
                    re *= shape
                    im *= shape

                    vis[r, f, 0] += re + im*1j
        else:
            raise ValueError("source_type must be "
                             "POINT or GAUSSIAN")

    return vis


@generated_jit(nopython=True, nogil=True, cache=True)
def wsclean_predict(uvw, lm, source_type, flux, coeffs,
                    log_poly, ref_freq, gauss_shape, frequency):
    arg_dtypes = tuple(np.dtype(a.dtype.name) for a
                       in (uvw, lm, flux, coeffs, ref_freq, frequency))
    dtype = np.result_type(np.complex64, *arg_dtypes)

    def impl(uvw, lm, source_type, flux, coeffs, log_poly,
             ref_freq, gauss_shape, frequency):
        spectrum = spectra(flux, coeffs, log_poly, ref_freq, frequency)
        return wsclean_predict_impl(uvw, lm, source_type, gauss_shape,
                                    frequency, spectrum, dtype)

    return impl


WSCLEAN_PREDICT_DOCS = DocstringTemplate("""
    Predict visibilities from a `WSClean sky model
    <https://sourceforge.net/p/wsclean/wiki/ComponentList/>`_.

    Parameters
    ----------
    uvw : $(array_type)
        UVW coordinates of shape :code:`(row, 3)`
    lm : $(array_type)
        Source LM coordinates of shape :code:`(source, 2)`, in radians.
        Derived from the ``Ra`` and ``Dec`` fields.
    source_type : $(array_type)
        Strings defining the source type of shape :code:`(source,)`.
        Should be either ``"POINT"`` or ``"GAUSSIAN"``.
        Contains the ``Type`` field.
    flux : $(array_type)
        Source flux of shape :code:`(source,)`.
        Contains the ``I`` field.
    coeffs : $(array_type)
        Source Polynomial coefficients of shape :code:`(source, coeffs)`.
        Contains the ``SpectralIndex`` field.
    log_poly : $(array_type)
        Source polynomial type of shape :code:`(source,)`.
        If True, logarithmic polynomials are used.
        If False, standard polynomials are used.
        Contains the ``LogarithmicSI`` field.
    ref_freq : $(array_type)
        Source Reference frequency of shape :code:`(source,)`.
        Contains the ``ReferenceFrequency`` field.
    gauss_shape : $(array_type)
        Gaussian shape parameters of shape :code:`(source, 3)`
        used when the corresponding ``source_type`` is ``"GAUSSIAN"``.
        The 3 components should contain the ``MajorAxis``, ``MinorAxis``
        and ``Orientation`` fields in radians, respectively.
    frequency : $(array_type)
        Frequency of shape :code:`(chan,)`.

    Returns
    -------
    visibilities : $(array_type)
        Complex visibilities of shape :code:`(row, chan, 1)`
""")

wsclean_predict.__doc__ = WSCLEAN_PREDICT_DOCS.substitute(
                            array_type=":class:`numpy.ndarray`")
