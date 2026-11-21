# -*- coding: utf-8 -*-

import numpy as np
from numba import get_num_threads, prange, set_num_threads, types
from numba.core.errors import TypingError

from africanus.constants import two_pi_over_c, c as lightspeed
from africanus.util.docs import DocstringTemplate
from africanus.util.numba import JIT_OPTIONS, overload, njit
from africanus.model.wsclean.spec_model import spectra

# numba cannot cache a function that calls get_num_threads or set_num_threads,
# nor any function calling it, and warns if asked to.
NOCACHE_JIT_OPTIONS = {**JIT_OPTIONS, "cache": False}


@njit(**JIT_OPTIONS)
def _source_terms(nsrc, lm, source_type, gauss_shape):
    """
    Computes the per-source terms that depend on neither row nor channel
    and checks the source types.
    """
    n1 = lm.dtype.type(1)

    n = np.empty(nsrc, lm.dtype)
    is_gauss = np.empty(nsrc, np.bool_)

    # Gaussian l-projection, m-projection and ratio.
    el = np.zeros(nsrc, gauss_shape.dtype)
    em = np.zeros(nsrc, gauss_shape.dtype)
    er = np.zeros(nsrc, np.float64)

    for s in range(nsrc):
        l = lm[s, 0]  # noqa
        m = lm[s, 1]
        n[s] = np.sqrt(n1 - l * l - m * m) - n1

        if source_type[s] == "POINT":
            is_gauss[s] = False
        elif source_type[s] == "GAUSSIAN":
            is_gauss[s] = True
            emaj, emin, angle = gauss_shape[s]

            # Convert to l-projection, m-projection, ratio
            el[s] = emaj * np.sin(angle)
            em[s] = emaj * np.cos(angle)
            er[s] = emin / (1.0 if emaj == 0.0 else emaj)
        else:
            raise ValueError("source_type must be POINT or GAUSSIAN")

    return n, is_gauss, el, em, er


@njit(**JIT_OPTIONS)
def _predict_row(
    r, uvw, lm, n, is_gauss, el, em, er, frequency, scaled_freq, spectrum, vis
):
    """Accumulates the visibilities of all sources into ``vis[r]``"""
    nsrc = n.shape[0]
    nchan = frequency.shape[0]

    u = uvw[r, 0]
    v = uvw[r, 1]
    w = uvw[r, 2]

    for s in range(nsrc):
        # Compute phase term
        real_phase = two_pi_over_c * (u * lm[s, 0] + v * lm[s, 1] + w * n[s])

        if is_gauss[s]:
            # Gaussian shape term bits
            u1 = (u * em[s] - v * el[s]) * er[s]
            v1 = u * el[s] + v * em[s]

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

                vis[r, f, 0] += re + im * 1j
        else:
            for f in range(nchan):
                p = real_phase * frequency[f]
                re = np.cos(p) * spectrum[s, f]
                im = np.sin(p) * spectrum[s, f]

                vis[r, f, 0] += re + im * 1j


@njit(**JIT_OPTIONS)
def _scaled_frequency(frequency):
    fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
    fwhminv = 1.0 / fwhm
    gauss_scale = fwhminv * np.sqrt(2.0) * np.pi / lightspeed

    return frequency * frequency.dtype.type(gauss_scale)


@njit(**JIT_OPTIONS)
def wsclean_predict_main(uvw, lm, source_type, gauss_shape, frequency, spectrum, dtype):
    nrow = uvw.shape[0]
    nchan = frequency.shape[0]
    ncorr = 1

    nsrc = spectrum.shape[0]

    scaled_freq = _scaled_frequency(frequency)
    n, is_gauss, el, em, er = _source_terms(nsrc, lm, source_type, gauss_shape)

    vis = np.zeros((nrow, nchan, ncorr), dtype=dtype)

    for r in range(nrow):
        _predict_row(
            r, uvw, lm, n, is_gauss, el, em, er, frequency, scaled_freq, spectrum, vis
        )

    return vis


@njit(parallel=True, **JIT_OPTIONS)
def _predict_rows_parallel(
    uvw, lm, n, is_gauss, el, em, er, frequency, scaled_freq, spectrum, vis
):
    # Each row only writes to vis[r], so rows can be predicted independently
    for r in prange(uvw.shape[0]):
        _predict_row(
            r, uvw, lm, n, is_gauss, el, em, er, frequency, scaled_freq, spectrum, vis
        )


# Sets the thread count, so it cannot be cached. It is kept separate
# from wsclean_predict_main so that the serial kernel can still be cached.
@njit(**NOCACHE_JIT_OPTIONS)
def wsclean_predict_parallel_main(
    uvw, lm, source_type, gauss_shape, frequency, spectrum, dtype, nthreads
):
    # The serial kernel never enters numba's threading layer. This matters
    # as the workqueue layer aborts if it is entered by concurrent callers,
    # such as the threads of a dask scheduler.
    if nthreads == 1:
        return wsclean_predict_main(
            uvw, lm, source_type, gauss_shape, frequency, spectrum, dtype
        )

    nrow = uvw.shape[0]
    nchan = frequency.shape[0]
    ncorr = 1

    nsrc = spectrum.shape[0]

    scaled_freq = _scaled_frequency(frequency)
    n, is_gauss, el, em, er = _source_terms(nsrc, lm, source_type, gauss_shape)

    vis = np.zeros((nrow, nchan, ncorr), dtype=dtype)

    # Nothing between setting the thread count and restoring it can raise:
    # set_num_threads validates nthreads before changing anything.
    prev_nthreads = get_num_threads()
    set_num_threads(nthreads)

    _predict_rows_parallel(
        uvw, lm, n, is_gauss, el, em, er, frequency, scaled_freq, spectrum, vis
    )

    set_num_threads(prev_nthreads)

    return vis


@njit(**JIT_OPTIONS)
def wsclean_predict(
    uvw,
    lm,
    source_type,
    flux,
    coeffs,
    log_poly,
    ref_freq,
    gauss_shape,
    frequency,
    nthreads=1,
):
    return wsclean_predict_impl(
        uvw,
        lm,
        source_type,
        flux,
        coeffs,
        log_poly,
        ref_freq,
        gauss_shape,
        frequency,
        nthreads,
    )


def wsclean_predict_impl(
    uvw,
    lm,
    source_type,
    flux,
    coeffs,
    log_poly,
    ref_freq,
    gauss_shape,
    frequency,
    nthreads,
):
    raise NotImplementedError


# The implementation is compiled into wsclean_predict, which is cached
# if the serial kernel is selected and cannot be cached otherwise.
@overload(wsclean_predict_impl, jit_options=NOCACHE_JIT_OPTIONS, prefer_literal=True)
def nb_wsclean_predict(
    uvw,
    lm,
    source_type,
    flux,
    coeffs,
    log_poly,
    ref_freq,
    gauss_shape,
    frequency,
    nthreads,
):
    if not isinstance(nthreads, types.Integer):
        raise TypingError(f"nthreads {nthreads} must be an integer")

    arg_dtypes = tuple(
        np.dtype(a.dtype.name) for a in (uvw, lm, flux, coeffs, ref_freq, frequency)
    )
    dtype = np.result_type(np.complex64, *arg_dtypes)

    if isinstance(nthreads, types.IntegerLiteral) and nthreads.literal_value == 1:
        # nthreads is known to be 1 during compilation, usually because it
        # was defaulted. Only the serial kernel is used so that this
        # specialisation of wsclean_predict can be cached.
        def impl(
            uvw,
            lm,
            source_type,
            flux,
            coeffs,
            log_poly,
            ref_freq,
            gauss_shape,
            frequency,
            nthreads,
        ):
            spectrum = spectra(flux, coeffs, log_poly, ref_freq, frequency)
            return wsclean_predict_main(
                uvw, lm, source_type, gauss_shape, frequency, spectrum, dtype
            )

    else:

        def impl(
            uvw,
            lm,
            source_type,
            flux,
            coeffs,
            log_poly,
            ref_freq,
            gauss_shape,
            frequency,
            nthreads,
        ):
            spectrum = spectra(flux, coeffs, log_poly, ref_freq, frequency)
            return wsclean_predict_parallel_main(
                uvw, lm, source_type, gauss_shape, frequency, spectrum, dtype, nthreads
            )

    return impl


WSCLEAN_PREDICT_DOCS = DocstringTemplate(
    """
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
    nthreads : int, optional
        Number of numba threads over which the prediction of rows
        is parallelised. Defaults to 1, which runs a serial kernel.
        Must be between 1 and :code:`numba.config.NUMBA_NUM_THREADS`.

    Returns
    -------
    visibilities : $(array_type)
        Complex visibilities of shape :code:`(row, chan, 1)`

    Notes
    -----
    * If ``nthreads`` is greater than 1, the numba thread count of the
      calling thread is set to ``nthreads`` for the duration of the
      prediction and restored to its previous value afterwards.
    * Calling this function with ``nthreads`` greater than 1 from multiple
      threads at the same time, for example from a multi-threaded dask
      scheduler or worker, uses ``nthreads`` threads for each calling thread.
      It also requires a thread-safe
      `numba threading layer
      <https://numba.readthedocs.io/en/stable/user/threading-layer.html>`_
      such as ``tbb`` or ``omp``, as the ``workqueue`` layer terminates the
      process on concurrent access.
    * numba cannot cache functions that set the thread count.
      The serial kernel is cached, but the parallel kernel is compiled
      the first time it is called in each process, and numba may warn
      that it cannot be cached.
"""
)

wsclean_predict.__doc__ = WSCLEAN_PREDICT_DOCS.substitute(
    array_type=":class:`numpy.ndarray`"
)
