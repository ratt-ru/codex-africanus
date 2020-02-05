# -*- coding: utf-8 -*-

from africanus.dft.kernels import im_to_vis_docs, vis_to_im_docs
from africanus.dft.kernels import im_to_vis as np_im_to_vis
from africanus.dft.kernels import vis_to_im as np_vis_to_im

from africanus.util.docs import doc_tuple_to_str
from africanus.util.requirements import requires_optional

import numpy as np

try:
    import dask.array as da
except ImportError as e:
    dask_import_error = e
else:
    dask_import_error = None


def _im_to_vis_wrapper(image, uvw, lm, frequency, convention, dtype_):
    return np_im_to_vis(image[0], uvw[0], lm[0][0], frequency,
                        convention=convention, dtype=dtype_)


@requires_optional('dask.array', dask_import_error)
def im_to_vis(image, uvw, lm, frequency,
              convention='fourier', dtype=np.complex128):
    """ Dask wrapper for im_to_vis function """
    if lm.chunks[0][0] != lm.shape[0]:
        raise ValueError("lm chunks must match lm shape "
                         "on first axis")
    if image.chunks[0][0] != image.shape[0]:
        raise ValueError("Image chunks must match image "
                         "shape on first axis")
    if image.chunks[0][0] != lm.chunks[0][0]:
        raise ValueError("Image chunks and lm chunks must "
                         "match on first axis")
    if image.chunks[1] != frequency.chunks[0]:
        raise ValueError("Image chunks must match frequency "
                         "chunks on second axis")
    return da.core.blockwise(_im_to_vis_wrapper, ("row", "chan", "corr"),
                             image, ("source", "chan", "corr"),
                             uvw, ("row", "(u,v,w)"),
                             lm, ("source", "(l,m)"),
                             frequency, ("chan",),
                             convention=convention,
                             dtype=dtype,
                             dtype_=dtype)


def _vis_to_im_wrapper(vis, uvw, lm, frequency, flags,
                       convention, dtype_):
    return np_vis_to_im(vis, uvw[0], lm[0],
                        frequency, flags,
                        convention=convention,
                        dtype=dtype_)[None, :]


@requires_optional('dask.array', dask_import_error)
def vis_to_im(vis, uvw, lm, frequency, flags,
              convention='fourier', dtype=np.float64):
    """ Dask wrapper for vis_to_im function """

    if vis.chunks[0] != uvw.chunks[0]:
        raise ValueError("Vis chunks and uvw chunks must "
                         "match on first axis")
    if vis.chunks[1] != frequency.chunks[0]:
        raise ValueError("Vis chunks must match frequency "
                         "chunks on second axis")
    if vis.chunks != flags.chunks:
        raise ValueError("Vis chunks must match flags "
                         "chunks on all axes")

    ims = da.core.blockwise(_vis_to_im_wrapper,
                            ("row", "source", "chan", "corr"),
                            vis, ("row", "chan", "corr"),
                            uvw, ("row", "(u,v,w)"),
                            lm, ("source", "(l,m)"),
                            frequency, ("chan",),
                            flags, ("row", "chan", "corr"),
                            adjust_chunks={"row": 1},
                            convention=convention,
                            dtype=dtype,
                            dtype_=dtype)

    return ims.sum(axis=0)


im_to_vis.__doc__ = doc_tuple_to_str(im_to_vis_docs,
                                     [(":class:`numpy.ndarray`",
                                         ":class:`dask.array.Array`")])

vis_to_im.__doc__ = doc_tuple_to_str(vis_to_im_docs,
                                     [(":class:`numpy.ndarray`",
                                         ":class:`dask.array.Array`")])
