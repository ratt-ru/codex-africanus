# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import wraps

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


@wraps(np_im_to_vis)
def _im_to_vis_wrapper(image, uvw, lm, frequency, dtype_):
    return np_im_to_vis(image[0], uvw[0], lm[0][0],
                        frequency, dtype=dtype_)


@requires_optional('dask.array', dask_import_error)
def im_to_vis(image, uvw, lm, frequency, dtype=np.complex128):
    """ Dask wrapper for phase_delay function """
    if lm.chunks[0][0] != lm.shape[0]:
        raise ValueError("lm chunks must match lm shape "
                         "on first axis")
    if image.chunks[0][0] != image.shape[0]:
        raise ValueError("Image chunks must match image "
                         "shape on first axis")
    if image.chunks[0][0] != lm.chunks[0][0]:
        raise ValueError("Image chunks and lm chunks must "
                         "match on first axis")
    return da.core.blockwise(_im_to_vis_wrapper, ("row", "chan"),
                             image, ("source", "chan"),
                             uvw, ("row", "(u,v,w)"),
                             lm, ("source", "(l,m)"),
                             frequency, ("chan",),
                             dtype=dtype,
                             dtype_=dtype)


@wraps(np_vis_to_im)
def _vis_to_im_wrapper(vis, uvw, lm, frequency, dtype_):
    return np_vis_to_im(vis, uvw[0], lm[0], frequency,
                        dtype=dtype_)[None, :]


@requires_optional('dask.array', dask_import_error)
def vis_to_im(vis, uvw, lm, frequency, dtype=np.float64):
    """ Dask wrapper for phase_delay_adjoint function """

    ims = da.core.blockwise(_vis_to_im_wrapper, ("row", "source", "chan"),
                            vis, ("row", "chan"),
                            uvw, ("row", "(u,v,w)"),
                            lm, ("source", "(l,m)"),
                            frequency, ("chan",),
                            adjust_chunks={"row": 1},
                            dtype=dtype,
                            dtype_=dtype)

    return ims.sum(axis=0)


im_to_vis.__doc__ = doc_tuple_to_str(im_to_vis_docs,
                                     [(":class:`numpy.ndarray`",
                                         ":class:`dask.array.Array`")])

vis_to_im.__doc__ = doc_tuple_to_str(vis_to_im_docs,
                                     [(":class:`numpy.ndarray`",
                                         ":class:`dask.array.Array`")])
