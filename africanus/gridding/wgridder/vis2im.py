# -*- coding: utf-8 -*-

try:
    from ducc0.wgridder import ms2dirty
except ImportError as e:
    ducc_import_error = e
else:
    ducc_import_error = None

import numpy as np
from africanus.util.docs import DocstringTemplate
from africanus.util.requirements import requires_optional


@requires_optional('ducc0.wgridder', ducc_import_error)
def _dirty_internal(uvw, freq, vis, freq_bin_idx, freq_bin_counts, nx, ny,
                    cell, weights, flag, celly, epsilon, nthreads,
                    do_wstacking):
    # adjust for chunking
    # need a copy here if using multiple row chunks
    freq_bin_idx2 = freq_bin_idx - freq_bin_idx.min()
    nband = freq_bin_idx.size
    if type(vis[0, 0]) == np.complex64:
        real_type = np.float32
    elif type(vis[0, 0]) == np.complex128:
        real_type = np.float64
    else:
        raise ValueError("Vis of incorrect type")
    # the extra dimension is required to allow for chunking over row
    dirty = np.zeros((1, nband, nx, ny), dtype=real_type)
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
        dirty[0, i] = ms2dirty(uvw=uvw, freq=freq[ind], ms=vis[:, ind],
                               wgt=wgt, npix_x=nx, npix_y=ny,
                               pixsize_x=cell, pixsize_y=celly,
                               nu=0, nv=0, epsilon=epsilon,
                               nthreads=nthreads, mask=mask,
                               do_wstacking=do_wstacking)
    return dirty


# This additional wrapper is required to allow the dask wrappers
# to chunk over row
@requires_optional('ducc0.wgridder', ducc_import_error)
def dirty(uvw, freq, vis, freq_bin_idx, freq_bin_counts, nx, ny, cell,
          weights=None, flag=None, celly=None, epsilon=None, nthreads=1,
          do_wstacking=True):
    # set precision
    if epsilon is None:
        if vis.dtype == np.complex128:
            epsilon = 1e-7
        elif vis.dtype == np.complex64:
            epsilon = 1e-5
        else:
            raise ValueError("vis of incorrect type")

    if celly is None:
        celly = cell

    if not nthreads:
        import multiprocessing
        nthreads = multiprocessing.cpu_count()

    dirty = _dirty_internal(uvw, freq, vis, freq_bin_idx, freq_bin_counts,
                            nx, ny, cell, weights, flag, celly,
                            epsilon, nthreads, do_wstacking)
    return dirty[0]


DIRTY_DOCS = DocstringTemplate(
    r"""
    Compute visibility to image mapping using ducc gridder i.e.

    .. math::


        I^D = R^\dagger \Sigma^{-1} V

    where :math:`R^\dagger` is an implicit gridding operator,
    :math:`V` denotes visibilities of shape :code:`(row, chan)` and
    :math:`I^D` is the dirty image of shape :code:`(band, nx, ny)`.

    The number of imaging bands :code:`(band)` must
    be less than or equal to the number of channels
    :code:`(chan)` at which the data were obtained.
    The mapping from :code:`(chan)` to :code:`(band)` is described
    by :code:`freq_bin_idx` and :code:`freq_bin_counts` as
    described below.

    Note that, if self adjoint gridding and degridding operators
    are required then :code:`weights` should be the square
    root of what is typically referred to as imaging weights and
    should also be passed into the degridder.
    In this case, the data needs to be pre-whitened.

    Parameters
    ----------
    uvw : $(array_type)
        uvw coordinates at which visibilities were
        obtained with shape :code:`(row, 3)`.
    freq : $(array_type)
        Observational frequencies of shape :code:`(chan,)`.
    vis : $(array_type)
        Visibilities of shape :code:`(row,chan)`.
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
    model : $(array_type)
        Dirty image corresponding to visibilities
        of shape :code:`(nband, nx, ny)`.
    """)

try:
    dirty.__doc__ = DIRTY_DOCS.substitute(
                        array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass
