# -*- coding: utf-8 -*-

import numpy as np
from africanus.util.docs import DocstringTemplate
from ducc0.wgridder import ms2dirty

def vis2im(uvw, freq, vis, weights, freq_bin_idx, freq_bin_counts,
           nx, ny, cellx, celly, nu, nv, epsilon, nthreads, do_wstacking):
    freq_bin_idx -= freq_bin_idx.min()  # adjust for chunking
    nband = freq_bin_idx.size
    dirty = np.zeros((nband, nx, ny), dtype=freq.dtype)
    for i in range(nband):
        I = slice(freq_bin_idx[i], freq_bin_idx[i] + freq_bin_counts[i])
        dirty[i] = ms2dirty(uvw=uvw, freq=freq[I], ms=vis[:, I],
                            wgt=weights[:, I], npix_x=nx, npix_y=ny,
                            pixsize_x=cellx, pixsize_y=celly,
                            nu=nu, nv=nv, epsilon=epsilon, nthreads=nthreads,
                            do_wstacking=do_wstacking)
    return dirty

VIS2IM_DOCS = DocstringTemplate(
    r"""
    Compute visibility to image mapping using ducc gridder i.e.

    .. math::


        I^D = R^\dagger \Sigma^{-1} V

    where :math:`R^\dagger` is an implicit gridding operator,
    :math:``V` denotes visibilities of shape :code:`(row, chan)` and 
    :math:`I^D` is the dirty image of shape :code:`(band, nx, ny)`.

    The number of imaging bands :code:`(band)` is has to
    be less than or equal to the number of channels 
    :code:`(chan)` at which the data were obtained.
    The mapping from :code:`(chan)` to :code:`(band)` is described
    by :code:`freq_bin_idx` and :code:`freq_bin_counts` as
    described below.

    Note that, if self adjoint gridding and degridding opeartors
    are required then :code:`weights` should actually be the square
    root of what is typically referred to as imaging weights and
    the same weights need to be passed into the degridder. 
    In this case, the data that are passed in need be pre-whitened.

    Parameters
    ----------
    uvw : $(array_type)
        uvw coordinates at which visibilities were
        obtained with shape :code:`(row, 3)`.
    freq : $(array_type)
        Observational frequencies of shape :code:`(chan,)`.
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
    model : $(array_type)
        Dirty image corresponding to visibilities
        of shape :code:`(nband, nx, ny)`.
    """)

try:
    vis2im.__doc__ = VIS2IM_DOCS.substitute(
                        array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass