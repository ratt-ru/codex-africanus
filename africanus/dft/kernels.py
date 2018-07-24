# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import numba
import numpy as np

from ..constants import minus_two_pi_over_c
from ..util.docs import doc_tuple_to_str


@numba.jit(nopython=True, nogil=True, cache=True)
def _im_to_vis_impl(image, uvw, lm, frequency, vis_of_im):
    # For each uvw coordinate
    for row in range(uvw.shape[0]):
        u, v, w = uvw[row]

        # For each source
        for source in range(lm.shape[0]):
            l, m = lm[source]
            n = np.sqrt(1.0 - l**2 - m**2) - 1.0

            # e^(-2*pi*(l*u + m*v + n*w)/c)
            real_phase = minus_two_pi_over_c * (l * u + m * v + n * w)

            # Multiple in frequency for each channel
            for chan in range(frequency.shape[0]):
                p = real_phase * frequency[chan]

                # Our phase input is purely imaginary
                # so we can can elide a call to exp
                # and just compute the cos and sin
                # @simon does this really make a difference?
                # I thought a complex exponential is evaluated
                # as a sum of sin and cos anyway
                vis_of_im[row, chan] += np.exp(p)*image[source, chan]/(n+1)

    return vis_of_im


@numba.jit(nopython=True, nogil=True, cache=True)
def _vis_to_im_impl(vis, uvw, lm, frequency, im_of_vis):
    # For each source
    for source in range(lm.shape[0]):
        l, m = lm[source]
        n = np.sqrt(1.0 - l ** 2 - m ** 2) - 1.0
        # For each uvw coordinate
        for row in range(uvw.shape[0]):
            u, v, w = uvw[row]

            # e^(-2*pi*(l*u + m*v + n*w)/c)
            real_phase = -minus_two_pi_over_c * (l * u + m * v + n * w)

            # Multiple in frequency for each channel
            for chan in range(frequency.shape[0]):
                p = real_phase * frequency[chan]

                im_of_vis[source,
                          chan] += (np.cos(p) * vis[row, chan].real -
                                    np.sin(p) * vis[row, chan].imag)/(n+1)
                # Note for the adjoint we don't need the imaginary part
                # and we can elide the call to exp

    return im_of_vis


def im_to_vis(image, uvw, lm, frequency, dtype=None):
    vis_of_im = np.zeros((uvw.shape[0], frequency.shape[0]),
                         dtype=np.complex128 if dtype is None else dtype)

    return _im_to_vis_impl(image, uvw, lm, frequency, vis_of_im)


def vis_to_im(vis, uvw, lm, frequency, dtype=None):
    im_of_vis = np.zeros((lm.shape[0], frequency.shape[0]),
                         dtype=np.float64 if dtype is None else dtype)

    return _vis_to_im_impl(vis, uvw, lm, frequency, im_of_vis)


_DFT_DOCSTRING = namedtuple(
    "_DFTDOCSTRING", ["preamble", "parameters", "returns"])

im_to_vis_docs = _DFT_DOCSTRING(
    preamble="""
    Computes the discrete image to visibility mapping of an ideal
    unpolarised interferometer :

    .. math::

        {\\Large \\sum_s \\frac{1}{n_s} e^{-2 \\pi i (u l_s + v m_s + w (n_s - 1))} \\cdot I_s }

    """,  # noqa

    parameters="""
    Parameters
    ----------

    image : :class:`numpy.ndarray`
        image of shape :code:`(source, chan)`
        The Stokes I intensity in each pixel (flatten 2D array per channel).
    uvw : :class:`numpy.ndarray`
        UVW coordinates of shape :code:`(row, 3)` with
        U, V and W components in the last dimension.
    lm : :class:`numpy.ndarray`
        LM coordinates of shape :code:`(source, 2)` with
        L and M components in the last dimension.
    frequency : :class:`numpy.ndarray`
        frequencies of shape :code:`(chan,)`
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


im_to_vis.__doc__ = doc_tuple_to_str(im_to_vis_docs)

vis_to_im_docs = _DFT_DOCSTRING(
    preamble="""
    Computes visibility to image mapping of an ideal
    unpolarised interferometer:

    .. math::

        {\\Large \\sum_k \\frac{1}{n} e^{ 2 \\pi i (u_k l + v_k m + w_k (n - 1))} \\cdot V_k}

    """,  # noqa

    parameters="""
    Parameters
    ----------

    vis : :class:`numpy.ndarray`
        visibilities of shape :code:`(row, chan)`
        The Stokes I visibilities of which to compute a dirty image
    uvw : :class:`numpy.ndarray`
        UVW coordinates of shape :code:`(row, 3)` with
        U, V and W components in the last dimension.
    lm : :class:`numpy.ndarray`
        LM coordinates of shape :code:`(source, 2)` with
        L and M components in the last dimension.
    frequency : :class:`numpy.ndarray`
        frequencies of shape :code:`(chan,)`
    dtype : np.dtype, optional
        Datatype of result. Should be either np.float32
        or np.float64. Defaults to np.float64
    """,

    returns="""
    Returns
    -------
    :class:`numpy.ndarray`
        float of shape :code:`(source, chan)`
    """
)


vis_to_im.__doc__ = doc_tuple_to_str(vis_to_im_docs)
