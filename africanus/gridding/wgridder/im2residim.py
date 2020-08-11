# -*- coding: utf-8 -*-

import numpy as np
from africanus.util.docs import DocstringTemplate
from ducc0.wgridder import dirty2ms, ms2dirty


def _im2residim_internal(uvw, freq, model, vis, weights, freq_bin_idx,
                        freq_bin_counts, cellx, celly, nu, nv, epsilon,
                        nthreads, do_wstacking):
    freq_bin_idx -= freq_bin_idx.min()  # adjust for chunking
    nband = freq_bin_idx.size
    _, nx, ny = model.shape
    # the extra dimension is required to allow for chunking over row
    residim = np.zeros((1, nband, nx, ny), dtype=model.dtype)
    for i in range(nband):
        ind = slice(freq_bin_idx[i], freq_bin_idx[i] + freq_bin_counts[i])
        residvis = vis[:, ind] - dirty2ms(uvw=uvw, freq=freq[ind],
                                          dirty=model[i], wgt=None,
                                          pixsize_x=cellx, pixsize_y=celly,
                                          nu=nu, nv=nv, epsilon=epsilon,
                                          nthreads=nthreads,
                                          do_wstacking=do_wstacking)
        residim[0, i] = ms2dirty(uvw=uvw, freq=freq[ind], ms=residvis,
                                 wgt=weights[:, ind], npix_x=nx, npix_y=ny,
                                 pixsize_x=cellx, pixsize_y=celly,
                                 nu=nu, nv=nv, epsilon=epsilon,
                                 nthreads=nthreads,
                                 do_wstacking=do_wstacking)
    return residim

# This additional wrapper is required to allow the dask wrappers
# to chunk over row


def im2residim(uvw, freq, model, vis, weights, freq_bin_idx, freq_bin_counts,
               cellx, celly, nu, nv, epsilon, nthreads, do_wstacking):
    residim = _im2residim_internal(uvw, freq, model, vis, weights,
                                   freq_bin_idx, freq_bin_counts,
                                   cellx, celly, nu, nv, epsilon,
                                   nthreads, do_wstacking)
    return residim[0]


IM2RESIDIM_DOCS = DocstringTemplate(
    r"""
    Compute residual image given a model and visibilities using ducc
    degridder i.e.

    .. math::


        I^R = R^\\dagger \\Sigma^{-1}(V - Rx)

    where :math:`R` is an implicit degridding operator, :math:``V`
    denotes visibilities of shape :code:`(row, chan)` and
    :math:`x` is the image of shape :code:`(band, nx, ny)`.

    The number of imaging bands :code:`(band)` is has to
    be less than or equal to the number of channels
    :code:`(chan)` at which the data were obtained.
    The mapping from :code:`(chan)` to :code:`(band)` is described
    by :code:`freq_bin_idx` and :code:`freq_bin_counts` as
    described below.

    Note that, if the gridding and degridding operators both apply
    the square root of the imaging weights then the visibilities
    that are passed in should be pre-whitened. In this case the
    function computes

    .. math::


        I^R = R^\\dagger \\Sigma^{-\\frac{1}{2}}(\\tilde{V}
              - \\Sigma^{-\\frac{1}{2}}Rx)

    which is identical to the above expression if
    :math:`\\tilde{V} = \\Sigma^{-\\frac{1}{2}}V`.

    Parameters
    ----------
    uvw : $(array_type)
        uvw coordinates at which visibilities were
        obtained with shape :code:`(row, 3)`.
    freq : $(array_type)
        Observational frequencies of shape :code:`(chan,)`.
    model : $(array_type)
        Model image to degrid of shape :code:`(nband, nx, ny)`.
    vis : $(array_type)
        Visibilities of shape :code:`(row,chan)`.
    weights : $(array_type)
        Imaging weights of shape :code:`(row, chan)`.
    freq_bin_idx : $(array_type)
        Starting indices of frequency bins for each imaging
        band of shape :code:`(band,)`.
    freq_bin_counts : $(array_type)
        The number of channels in each imaging band of shape :code:`(band,)`.
    cellx : float
        The cell size of a pixel along the :math:`x` direction in radians.
    celly : float
        The cell size of a pixel along the :math:`y` direction in radians.
    nu : int
        The number of pixels in the padded grid along the :math:`x` direction.
    nv : int
        The number of pixels in the padded grid along the :math:`y` direction.
    epsilon : float
        The precision of the degridding operator with respect to the
        direct Fourier transform.
    nthreads : int
        The number of threads to use.
    do_wstacking : bool
        Whether to correct for the w-term or not.
    complex_type : np.dtype
        The data type of output visibilities.

    Returns
    -------
    vis : $(array_type)
        Visibilities corresponding to :code:`model` of shape
        :code:`(row,chan)`.
    """)

try:
    im2residim.__doc__ = IM2RESIDIM_DOCS.substitute(
                        array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass
