# -*- coding: utf-8 -*-

import numpy as np
from africanus.util.docs import DocstringTemplate
from ducc0.wgridder import dirty2ms


def im2vis(uvw, freq, model, weights, freq_bin_idx, freq_bin_counts,
           cellx, celly, nu, nv, epsilon, nthreads, do_wstacking,
           complex_type):
    # adjust for chunking
    # need a copy here if using multiple row chunks
    freq_bin_idx2 = freq_bin_idx - freq_bin_idx.min()  
    nband = freq_bin_idx.size
    nrow = uvw.shape[0]
    nchan = freq.size
    vis = np.zeros((nrow, nchan), dtype=complex_type)
    for i in range(nband):
        ind = slice(freq_bin_idx2[i], freq_bin_idx2[i] + freq_bin_counts[i])
        if weights is not None:
            wgt = weights[:, ind]
        else:
            wgt = None
        vis[:, ind] = dirty2ms(uvw=uvw, freq=freq[ind], dirty=model[i],
                               wgt=wgt, pixsize_x=cellx, pixsize_y=celly,
                               nu=nu, nv=nv, epsilon=epsilon,
                               nthreads=nthreads, do_wstacking=do_wstacking)
    return vis


IM2VIS_DOCS = DocstringTemplate(
    r"""
    Compute image to visibility mapping using ducc degridder i.e.

    .. math::


        V = Rx

    where :math:`R` is an implicit degridding operator, :math:``V`
    denotes visibilities of shape :code:`(row, chan)` and
    :math:`x` is the image of shape :code:`(band, nx, ny)`.

    The number of imaging bands :code:`(band)` is has to
    be less than or equal to the number of channels
    :code:`(chan)` at which the data were obtained.
    The mapping from :code:`(chan)` to :code:`(band)` is described
    by :code:`freq_bin_idx` and :code:`freq_bin_counts` as
    described below.

    There is an option to provide weights during degridding.
    This option is made available to allow for the possibility
    of having self adjoint gridding and degridding operators.
    In this case :code:`weights` should actually be the square
    root of what is typically referred to as imaging weights.
    In this case the degridder computes the whitened model
    visibilities i.e.

    .. math::


        V = \Sigma^{-\\frac{1}{2}} R x

    where :math:`\Sigma` refers to the inverse of the weights
    (i.e. the data covariance matrix when using natural weighting).

    Parameters
    ----------
    uvw : $(array_type)
        uvw coordinates at which visibilities were
        obtained with shape :code:`(row, 3)`.
    freq : $(array_type)
        Observational frequencies of shape :code:`(chan,)`.
    model : $(array_type)
        Model image to degrid of shape :code:`(nband, nx, ny)`.
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
    im2vis.__doc__ = IM2VIS_DOCS.substitute(
                        array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass
