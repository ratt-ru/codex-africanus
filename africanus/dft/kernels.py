# -*- coding: utf-8 -*-


from africanus.util.numba import is_numba_type_none, generated_jit
from africanus.util.docs import doc_tuple_to_str
from collections import namedtuple

import numba
import numpy as np

from africanus.constants import minus_two_pi_over_c, two_pi_over_c


@generated_jit(nopython=True, nogil=True, cache=True)
def im_to_vis(image, uvw, lm, frequency,
              convention='fourier', dtype=None):
    # Infer complex output dtype if none provided
    if is_numba_type_none(dtype):
        out_dtype = np.result_type(np.complex64,
                                   *(np.dtype(a.dtype.name) for a in
                                     (image, uvw, lm, frequency)))
    else:
        out_dtype = dtype.dtype

    def impl(image, uvw, lm, frequency,
             convention='fourier', dtype=None):
        if convention == 'fourier':
            constant = minus_two_pi_over_c
        elif convention == 'casa':
            constant = two_pi_over_c
        else:
            raise ValueError("convention not in ('fourier', 'casa')")

        nrows = uvw.shape[0]
        nsrc = lm.shape[0]
        nchan = frequency.shape[0]
        ncorr = image.shape[-1]
        vis_of_im = np.zeros((nrows, nchan, ncorr), dtype=out_dtype)

        # For each uvw coordinate
        for r in range(nrows):
            u, v, w = uvw[r]

            # For each source
            for s in range(nsrc):
                l, m = lm[s]
                n = np.sqrt(1.0 - l**2 - m**2) - 1.0

                # e^(-2*pi*(l*u + m*v + n*w)/c)
                real_phase = constant * (l * u + m * v + n * w)

                # Multiple in frequency for each channel
                for nu in range(nchan):
                    p = real_phase * frequency[nu] * 1.0j

                    for c in range(ncorr):
                        if image[s, nu, c]:
                            vis_of_im[r, nu, c] += np.exp(p)*image[s, nu, c]

        return vis_of_im

    return impl


@generated_jit(nopython=True, nogil=True, cache=True)
def vis_to_im(vis, uvw, lm, frequency, flags,
              convention='fourier', dtype=None):
    # Infer output dtype if none provided
    if is_numba_type_none(dtype):
        # Support both real and complex visibilities...
        if isinstance(vis.dtype, numba.types.scalars.Complex):
            vis_comp_dtype = np.dtype(vis.dtype.underlying_float.name)
        else:
            vis_comp_dtype = np.dtype(vis.dtype.name)

        out_dtype = np.result_type(vis_comp_dtype,
                                   *(np.dtype(a.dtype.name) for a in
                                     (uvw, lm, frequency)))
    else:
        if isinstance(dtype, numba.types.scalars.Complex):
            raise TypeError("dtype must be complex")

        out_dtype = dtype.dtype

    assert np.shape(vis) == np.shape(flags)

    def impl(vis, uvw, lm, frequency, flags,
             convention='fourier', dtype=None):
        nrows = uvw.shape[0]
        nsrc = lm.shape[0]
        nchan = frequency.shape[0]
        ncorr = vis.shape[-1]

        if convention == 'fourier':
            constant = two_pi_over_c
        elif convention == 'casa':
            constant = minus_two_pi_over_c
        else:
            raise ValueError("convention not in ('fourier', 'casa')")

        im_of_vis = np.zeros((nsrc, nchan, ncorr), dtype=out_dtype)

        # For each source
        for s in range(nsrc):
            l, m = lm[s]
            n = np.sqrt(1.0 - l ** 2 - m ** 2) - 1.0
            # For each uvw coordinate
            for r in range(nrows):
                u, v, w = uvw[r]

                # e^(-2*pi*(l*u + m*v + n*w)/c)
                real_phase = constant * (l * u + m * v + n * w)

                # Multiple in frequency for each channel
                for nu in range(nchan):
                    p = real_phase * frequency[nu]

                    # do not compute if any of the correlations
                    # are flagged (complicates uncertainties)
                    if np.any(flags[r, nu]):
                        continue

                    for c in range(ncorr):
                        # elide the call to exp since result is real
                        im_of_vis[s, nu, c] += (np.cos(p) *
                                                vis[r, nu, c].real -
                                                np.sin(p) *
                                                vis[r, nu, c].imag)

        return im_of_vis

    return impl


_DFT_DOCSTRING = namedtuple(
    "_DFTDOCSTRING", ["preamble", "parameters", "returns"])

im_to_vis_docs = _DFT_DOCSTRING(
    preamble="""
    Computes the discrete image to visibility mapping
    of an ideal interferometer:

    .. math::

        {\\Large \\sum_s e^{-2 \\pi i (u l_s + v m_s + w (n_s - 1))} \\cdot I_s }

    """,  # noqa

    parameters=r"""
    Parameters
    ----------

    image : :class:`numpy.ndarray`
        image of shape :code:`(source, chan, corr)`
        The brighness matrix in each pixel (flatten 2D array
        per channel and corr). Note not Stokes terms
    uvw : :class:`numpy.ndarray`
        uvw coordinates of shape :code:`(row, 3)` with
        u, v and w components in the last dimension.
    lm : :class:`numpy.ndarray`
        lm coordinates of shape :code:`(source, 2)` with
        l and m components in the last dimension.
    frequency : :class:`numpy.ndarray`
        frequencies of shape :code:`(chan,)`
    convention : {'fourier', 'casa'}
        Uses the :math:`e^{-2 \pi \mathit{i}}` sign convention
        if ``fourier`` and :math:`e^{2 \pi \mathit{i}}` if
        ``casa``.
    dtype : np.dtype, optional
        Datatype of result. Should be either np.complex64 or np.complex128.
        If ``None``, :func:`numpy.result_type` is used to infer the data type
        from the inputs.
    """,

    returns="""
    Returns
    -------
    visibilties : :class:`numpy.ndarray`
        complex of shape :code:`(row, chan, corr)`
    """
)


im_to_vis.__doc__ = doc_tuple_to_str(im_to_vis_docs)

vis_to_im_docs = _DFT_DOCSTRING(
    preamble="""
    Computes visibility to image mapping
    of an ideal interferometer:

    .. math::

        {\\Large \\sum_k e^{ 2 \\pi i (u_k l + v_k m + w_k (n - 1))} \\cdot V_k}

    """,  # noqa

    parameters=r"""
    Parameters
    ----------

    vis : :class:`numpy.ndarray`
        visibilities of shape :code:`(row, chan, corr)`
        Visibilities corresponding to brightness terms.
        Note the dirty images produced do not necessarily
        correspond to Stokes terms and need to be converted.
    uvw : :class:`numpy.ndarray`
        uvw coordinates of shape :code:`(row, 3)` with
        u, v and w components in the last dimension.
    lm : :class:`numpy.ndarray`
        lm coordinates of shape :code:`(source, 2)` with
        l and m components in the last dimension.
    frequency : :class:`numpy.ndarray`
        frequencies of shape :code:`(chan,)`
    flags : :class:`numpy.ndarray`
        Boolean array of shape :code:`(row, chan, corr)`
        Note that if one correlation is flagged we discard
        all of them otherwise we end up irretrievably
        mixing Stokes terms.
    convention : {'fourier', 'casa'}
        Uses the :math:`e^{-2 \pi \mathit{i}}` sign convention
        if ``fourier`` and :math:`e^{2 \pi \mathit{i}}` if
        ``casa``.
    dtype : np.dtype, optional
        Datatype of result. Should be either np.float32 or np.float64.
        If ``None``, :func:`numpy.result_type` is used to infer the data type
        from the inputs.
    """,

    returns="""
    Returns
    -------
    image : :class:`numpy.ndarray`
        float of shape :code:`(source, chan, corr)`
    """
)


vis_to_im.__doc__ = doc_tuple_to_str(vis_to_im_docs)
