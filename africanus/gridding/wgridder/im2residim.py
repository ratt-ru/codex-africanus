# -*- coding: utf-8 -*-

try:
    from ducc0.wgridder import dirty2ms, ms2dirty
except ImportError as e:
    ducc_import_error = e
else:
    ducc_import_error = None

import numpy as np
from africanus.util.docs import DocstringTemplate
from africanus.util.requirements import requires_optional


@requires_optional('ducc0.wgridder', ducc_import_error)
def _residual_internal(uvw, freq, image, vis, freq_bin_idx, freq_bin_counts,
                       cell, weights, flag, celly, epsilon, nthreads,
                       do_wstacking):

    # adjust for chunking
    # need a copy here if using multiple row chunks
    freq_bin_idx2 = freq_bin_idx - freq_bin_idx.min()
    nband = freq_bin_idx.size
    _, nx, ny = image.shape
    # the extra dimension is required to allow for chunking over row
    residim = np.zeros((1, nband, nx, ny), dtype=image.dtype)
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
        residvis = vis[:, ind] - dirty2ms(uvw=uvw, freq=freq[ind],
                                          dirty=image[i], wgt=None,
                                          pixsize_x=cell, pixsize_y=celly,
                                          nu=0, nv=0, epsilon=epsilon,
                                          nthreads=nthreads, mask=mask,
                                          do_wstacking=do_wstacking)
        residim[0, i] = ms2dirty(uvw=uvw, freq=freq[ind], ms=residvis,
                                 wgt=wgt, npix_x=nx, npix_y=ny,
                                 pixsize_x=cell, pixsize_y=celly,
                                 nu=0, nv=0, epsilon=epsilon,
                                 nthreads=nthreads, mask=mask,
                                 do_wstacking=do_wstacking)
    return residim


# This additional wrapper is required to allow the dask wrappers
# to chunk over row
@requires_optional('ducc0.wgridder', ducc_import_error)
def residual(uvw, freq, image, vis, freq_bin_idx, freq_bin_counts, cell,
             weights=None, flag=None, celly=None, epsilon=None, nthreads=1,
             do_wstacking=True):
    # set precision
    if epsilon is None:
        if image.dtype == np.float64:
            epsilon = 1e-7
        elif image.dtype == np.float32:
            epsilon = 1e-5
        else:
            raise ValueError("Model of incorrect type")

    if celly is None:
        celly = cell

    if not nthreads:
        import multiprocessing
        nthreads = multiprocessing.cpu_count()

    residim = _residual_internal(uvw, freq, image, vis, freq_bin_idx,
                                 freq_bin_counts, cell, weights, flag,
                                 celly, epsilon, nthreads, do_wstacking)
    return residim[0]


RESIDUAL_DOCS = DocstringTemplate(
    r"""
    Compute residual image given a model and visibilities using ducc
    degridder i.e.

    .. math::


        I^R = R^\dagger \Sigma^{-1}(V - Rx)

    where :math:`R` is an implicit degridding operator, :math:`V`
    denotes visibilities of shape :code:`(row, chan)` and
    :math:`x` is the image of shape :code:`(band, nx, ny)`.

    The number of imaging bands :code:`(band)` must
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


        I^R = R^\dagger \Sigma^{-\frac{1}{2}}(\tilde{V}
              - \Sigma^{-\frac{1}{2}}Rx)

    which is identical to the above expression if
    :math:`\tilde{V} = \Sigma^{-\frac{1}{2}}V`.

    Parameters
    ----------
    uvw : $(array_type)
        uvw coordinates at which visibilities were
        obtained with shape :code:`(row, 3)`.
    freq : $(array_type)
        Observational frequencies of shape :code:`(chan,)`.
    model : $(array_type)
        Model image to degrid of shape :code:`(band, nx, ny)`.
    vis : $(array_type)
        Visibilities of shape :code:`(row,chan)`.
    weights : $(array_type)
        Imaging weights of shape :code:`(row, chan)`.
    freq_bin_idx : $(array_type)
        Starting indices of frequency bins for each imaging
        band of shape :code:`(band,)`.
    freq_bin_counts : $(array_type)
        The number of channels in each imaging band of shape :code:`(band,)`.
    cell : float
        The cell size of a pixel along the :math:`x` direction in radians.
    flag: $(array_type), optional
        Flags of shape :code:`(row,chan)`. Will only process visibilities
        for which flag!=0
    celly : float, optional
        The cell size of a pixel along the :math:`y` direction in radians.
        By default same as cell size along :math:`x` direction.
    nu : int, optional
        The number of pixels in the padded grid along the :math:`x` direction.
        Chosen automatically by default.
    nv : int, optional
        The number of pixels in the padded grid along the :math:`y` direction.
        Chosen automatically by default.
    epsilon : float, optional
        The precision of the gridder with respect to the direct Fourier
        transform. By deafult, this is set to :code:`1e-5` for single
        precision and :code:`1e-7` for double precision.
    nthreads : int, optional
        The number of threads to use. Defaults to one.
    do_wstacking : bool, optional
        Whether to correct for the w-term or not. Defaults to True

    Returns
    -------
    residual : $(array_type)
        Residual image corresponding to :code:`model` of shape
        :code:`(band, nx, ny)`.
    """)

try:
    residual.__doc__ = RESIDUAL_DOCS.substitute(
                        array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass
