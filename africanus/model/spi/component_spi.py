#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np

from africanus.util.docs import DocstringTemplate
from africanus.util.numba import jit


@jit(nopython=True, nogil=True, cache=True)
def _fit_spi_components_impl(data, weights, freqs, freq0, out,
                             jac, beam, ncomps, nfreqs,
                             tol, maxiter, mindet):
    w = freqs/freq0
    dof = np.maximum(w.size - 2, 1)
    for comp in range(ncomps):
        eps = 1.0
        k = 0
        alphak = out[0, comp]
        i0k = out[2, comp]
        b = beam[comp]
        while eps > tol and k < maxiter:
            alphap = alphak
            i0p = i0k
            jac[1, :] = b*w**alphak
            model = i0k*jac[1, :]
            jac[0, :] = model * np.log(w)
            residual = data[comp] - model
            lik = 0.0
            hess00 = 0.0
            hess01 = 0.0
            hess11 = 0.0
            jr0 = 0.0
            jr1 = 0.0
            for v in range(nfreqs):
                lik += residual[v] * weights[v] * residual[v]
                jr0 += jac[0, v] * weights[v] * residual[v]
                jr1 += jac[1, v] * weights[v] * residual[v]
                hess00 += jac[0, v] * weights[v] * jac[0, v]
                hess01 += jac[0, v] * weights[v] * jac[1, v]
                hess11 += jac[1, v] * weights[v] * jac[1, v]
            det = np.maximum(hess00 * hess11 - hess01**2, mindet)
            alphak = alphap + (hess11 * jr0 - hess01 * jr1)/det
            i0k = i0p + (-hess01 * jr0 + hess00 * jr1)/det
            eps = np.maximum(np.abs(alphak - alphap), np.abs(i0k - i0p))
            k += 1
        if k == maxiter:
            print("Warning - max iterations exceeded for component ", comp)
        out[0, comp] = alphak
        out[1, comp] = hess11/det * lik/dof
        out[2, comp] = i0k
        out[3, comp] = hess00/det * lik/dof
    return out


def fit_spi_components(data, weights, freqs, freq0,
                       alphai=None, I0i=None, beam=None,
                       tol=1e-4, maxiter=100):
    ncomps, nfreqs = data.shape
    if beam is None:
        beam = np.ones(data.shape, data.dtype)
    jac = np.zeros((2, nfreqs), dtype=data.dtype)
    out = np.zeros((4, ncomps), dtype=data.dtype)
    if alphai is not None:
        out[0, :] = alphai
    else:
        out[0, :] = -0.7 * np.ones(ncomps, dtype=data.dtype)
    if I0i is not None:
        out[2, :] = I0i
    else:
        tmp = np.abs(freqs - freq0)
        ref_freq_idx = np.argwhere(tmp == tmp.min()).squeeze()
        if np.size(ref_freq_idx) > 1:
            ref_freq_idx = ref_freq_idx.min()
        out[2, :] = data[:, ref_freq_idx]/beam[:, ref_freq_idx]
    if data.dtype == np.float64:
        mindet = 1e-12
    elif data.dtype == np.float32:
        mindet = 1e-5
    else:
        raise ValueError("Unsupported data type. Must be float32 of float64.")

    return _fit_spi_components_impl(data, weights, freqs, freq0, out,
                                    jac, beam, ncomps, nfreqs,
                                    tol, maxiter, mindet)


SPI_DOCSTRING = DocstringTemplate(
    r"""
    Computes the spectral indices and the intensity
    at the reference frequency of a spectral index model:

    .. math::

        I(\nu) = A(\nu) I(\nu_0) \left( \frac{\nu}{\nu_0} \right) ^ \alpha

    where :math:`I(\nu)` is the apparent source spectrum,
    :math:`A(\nu)` is the beam model for each component as a function of
    frequency.

    Parameters
    ----------

    data : $(array_type)
        array of shape :code:`(comps, chan)`
        The noisy data as a function of frequency.
    weights : $(array_type)
        array of shape :code:`(chan,)`
        Inverse of variance on each frequency axis.
    freqs : $(array_type)
        frequencies of shape :code:`(chan,)`
    freq0 : float
        Reference frequency
    alphai : $(array_type), optional
        array of shape :code:`(comps,)`
        Initial guess for the alphas. Defaults
        to -0.7.
    I0i : $(array_type), optional
        array of shape :code:`(comps,)`
        Initial guess for the intensities at the
        reference frequency. Defaults to 1.0.
    beam_comps : $(array_type), optional
        array of shape :code:`(comps, chan)`
        Power beam for each component as a function of frequency.
    tol : float, optional
        Solver absolute tolerance (optional).
        Defaults to 1e-6.
    maxiter : int, optional
        Solver maximum iterations (optional).
        Defaults to 100.
    dtype : np.dtype, optional
        Datatype of result. Should be either np.float32
        or np.float64. Defaults to np.float64.

    Returns
    -------
    out : $(array_type)
        array of shape :code:`(4, comps)`
        The fitted components arranged
        as [alphas, alphavars, I0s, I0vars]
    """)

try:
    fit_spi_components.__doc__ = SPI_DOCSTRING.substitute(
                            array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass
