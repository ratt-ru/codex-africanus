# -*- coding: utf-8 -*-

try:
    from ducc0.wgridder import dirty2ms
except ImportError as e:
    ducc_import_error = e
else:
    ducc_import_error = None

import numpy as np
from africanus.util.docs import DocstringTemplate
from africanus.util.requirements import requires_optional


@requires_optional('ducc0.wgridder', ducc_import_error)
def _model_internal(uvw, freq, image, freq_bin_idx, freq_bin_counts, cell,
                    weights, flag, celly, epsilon, nthreads, do_wstacking):
    # adjust for chunking
    # need a copy here if using multiple row chunks
    freq_bin_idx2 = freq_bin_idx - freq_bin_idx.min()
    nband, nx, ny = image.shape
    nrow = uvw.shape[0]
    nchan = freq.size
    vis = np.zeros((nrow, nchan), dtype=np.result_type(image, np.complex64))
    for i in range(nband):
        ind = slice(freq_bin_idx2[i], freq_bin_idx2[i] + freq_bin_counts[i])
        if weights is not None:
            wgt = weights[:, ind]
        else:
            wgt = None
        if flag is not None:
            mask = flag[:, ind]
        else:
            mask = None
        vis[:, ind] = dirty2ms(uvw=uvw, freq=freq[ind], dirty=image[i],
                               wgt=wgt, pixsize_x=cell, pixsize_y=celly,
                               nu=0, nv=0, epsilon=epsilon, mask=mask,
                               nthreads=nthreads, do_wstacking=do_wstacking)
    return vis


@requires_optional('ducc0.wgridder', ducc_import_error)
def model(uvw, freq, image, freq_bin_idx, freq_bin_counts, cell, weights=None,
          flag=None, celly=None, epsilon=1e-5, nthreads=1, do_wstacking=True):

    if celly is None:
        celly = cell

    if not nthreads:
        import multiprocessing
        nthreads = multiprocessing.cpu_count()

    return _model_internal(uvw, freq, image, freq_bin_idx, freq_bin_counts,
                           cell, weights, flag, celly, epsilon, nthreads,
                           do_wstacking)


MODEL_DOCS = DocstringTemplate(
    r"""
    Compute image to visibility mapping using ducc degridder i.e.

    .. math::


        V = Rx

    where :math:`R` is an implicit degridding operator, :math:`V`
    denotes visibilities of shape :code:`(row, chan)` and
    :math:`x` is the image of shape :code:`(band, nx, ny)`.

    The number of imaging bands :code:`(band)` has to
    be less than or equal to the number of channels
    :code:`(chan)` at which the data were obtained.
    The mapping from :code:`(chan)` to :code:`(band)` is described
    by :code:`freq_bin_idx` and :code:`freq_bin_counts` as
    described below.

    There is an option to provide weights during degridding
    to cater for self adjoint gridding and degridding operators.
    In this case :code:`weights` should actually be the square
    root of what is typically referred to as imaging weights.
    In this case the degridder computes the whitened model
    visibilities i.e.

    .. math::


        V = \Sigma^{-\frac{1}{2}} R x

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
    freq_bin_idx : $(array_type)
        Starting indices of frequency bins for each imaging
        band of shape :code:`(band,)`.
    freq_bin_counts : $(array_type)
        The number of channels in each imaging band of shape :code:`(band,)`.
    cell : float
        The cell size of a pixel along the :math:`x` direction in radians.
    weights : $(array_type), optional
        Imaging weights of shape :code:`(row, chan)`.
    flag: $(array_type), optional
        Flags of shape :code:`(row,chan)`. Will only process visibilities
        for which flag!=0
    celly : float, optional
        The cell size of a pixel along the :math:`y` direction in radians.
        By default same as cell size along :math:`x` direction.
    epsilon : float, optional
        The precision of the gridder with respect to the direct Fourier
        transform. By deafult, this is set to :code:`1e-5` for single
        precision and :code:`1e-7` for double precision.
    nthreads : int, optional
        The number of threads to use. Defaults to one.
        If set to zero will use all available cores.
    do_wstacking : bool, optional
        Whether to correct for the w-term or not. Defaults to True

    Returns
    -------
    vis : $(array_type)
        Visibilities corresponding to :code:`model` of shape
        :code:`(row,chan)`.
    """)

try:
    model.__doc__ = MODEL_DOCS.substitute(
                        array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass
