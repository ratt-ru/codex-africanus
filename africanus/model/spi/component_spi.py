#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
from ...util.docs import doc_tuple_to_str

import numba
import numpy as np


@numba.jit(nopython=True, nogil=True, cache=True)
def _fit_spi_components_impl(data, weights, freqs, freq0,
                             alphas, alphavars, i0s, i0vars, jac,
                             ncomps, nfreqs, tol, maxiter):
    w = freqs/freq0
    for comp in range(ncomps):
        eps = 1.0
        k = 0
        alphak = alphas[comp]
        i0k = i0s[comp]
        while eps > tol and k < maxiter:
            alphap = alphak
            i0p = i0k
            jac[1, :] = w**alphak
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
            det = hess00 * hess11 - hess01**2
            alphak = alphap + (hess11 * jr0 - hess01 * jr1)/det
            i0k = i0p + (-hess01 * jr0 + hess00 * jr1)/det
            eps = np.maximum(np.abs(alphak - alphap), np.abs(i0k - i0p))
            k += 1
        if k >= maxiter:
            print("Warning - max iterations exceeded for component ", comp)
        alphas[comp] = alphak
        alphavars[comp] = hess11/det
        i0s[comp] = i0k
        i0vars[comp] = hess00/det
    return alphas, alphavars, i0s, i0vars

def fit_spi_components(data, weights, freqs, freq0,
                       alphai=None, I0i=None, tol=1e-6,
                       maxiter=100, dtype=np.float64):
    ncomps, nfreqs = data.shape
    jac = np.zeros((2, nfreqs), dtype=dtype)
    if alphai is not None:
        alphas = alphai
    else:
        alphas = -0.7 * np.ones(ncomps, dtype=dtype)
    alphavars = np.zeros(ncomps, dtype=dtype)
    if I0i is not None:
        I0s = I0i
    else:
        I0s = np.ones(ncomps, dtype=dtype)
    I0vars = np.zeros(ncomps, dtype=dtype)
    return _fit_spi_components_impl(data, weights, freqs, freq0,
                                    alphas, alphavars, I0s, I0vars, jac,
                                    ncomps, nfreqs, tol, maxiter)

_SPI_DOCSTRING = namedtuple(
    "_SPIDOCSTRING", ["preamble", "parameters", "returns"])

im_to_vis_docs = _SPI_DOCSTRING(
    preamble="""
    Computes the spectral indices and the intensity 
    at the reference frequency of a spectral index model:

    .. math::

        {I(\\nu) = I_0(\\nu_0) \\left( \\frac{\\nu}{\\nu_0} \\right) ^ \\alpha }

    """,  # noqa

    parameters="""
    Parameters
    ----------

    data : :class:`numpy.ndarray`
        array of shape :code:`(comps, chan)`
        The noisy data as a function of frequency.
    weights : :class:`numpy.ndarray`
        array of shape :code:`(chan)`
        Inverse of variance on each frequency axis.
    freqs : :class:`numpy.ndarray`
        frequencies of shape :code:`(chan,)`
    freq0 : :float:
        Reference frequency
    alphai : :class:`numpy.ndarray`, optional
        array of shape :code:`(comps)`
        Initial guess for the alphas
    I0i : :class:`numpy.ndarray`, optional
        array of shape :code:`(comps)`
        Initial guess for the intensities at the 
        reference frequency
    tol : np.float, optional
        solver absolute tolerance (optional)
    maxiter : np.int, optional
        solver maximum iterations (optional)
    dtype : np.dtype, optional
        Datatype of result. Should be either np.complex64
        or np.complex128. Defaults to np.complex128
    """,

    returns="""
    Returns
    -------
    :class:`numpy.ndarray`
        complex of shape :code:`(row, chan)`
    """
)


fit_spi_components.__doc__ = doc_tuple_to_str(im_to_vis_docs)
