# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .kernels import im_to_vis_docs, vis_to_im_docs
from .kernels import im_to_vis as np_im_to_vis
from .kernels import vis_to_im as np_vis_to_im

from ..util.docs import on_rtd, doc_tuple_to_str, mod_docs
from ..util.requirements import have_packages, MissingPackageException

_package_requirements = ('dask.array',)
have_requirements = have_packages(*_package_requirements)

if not have_requirements or on_rtd():
    def im_to_vis(image, uvw, lm, frequency, dtype=None):
        raise MissingPackageException(*_package_requirements)

    def vis_to_im(vis, uvw, lm, frequency, dtype=None):
        raise MissingPackageException(*_package_requirements)
else:
    import numpy as np
    import dask.array as da

    def im_to_vis(image, uvw, lm, frequency, dtype=np.complex128):
        """ Dask wrapper for phase_delay function """
        def _wrapper(image, uvw, lm, frequency, dtype_):
            return np_im_to_vis(image[0], uvw[0], lm[0], frequency, dtype=dtype_)

        return da.core.atop(_wrapper, ("row", "chan"),
                            uvw, ("row", "(u,v,w)"),
                            lm, ("source", "(l,m)"),
                            frequency, ("chan",),
                            dtype=dtype,
                            dtype_=dtype)

    def vis_to_im(vis, uvw, lm, frequency, dtype=np.complex128):
        """ Dask wrapper for phase_delay_adjoint function """
        def _wrapper(vis, uvw, lm, frequency, dtype_):
            return np_vis_to_im(vis[0], uvw[0], lm[0], frequency, dtype=dtype_)

        return da.core.atop(_wrapper, ("source", "chan"),
                            uvw, ("row", "(u,v,w)"),
                            lm, ("source", "(l,m)"),
                            frequency, ("chan",),
                            dtype=dtype,
                            dtype_=dtype)


im_to_vis.__doc__ = doc_tuple_to_str(im_to_vis_docs,
                                       [(":class:`numpy.ndarray`",
                                         ":class:`dask.array.Array`")])

vis_to_im.__doc__ = doc_tuple_to_str(vis_to_im_docs,
                                       [(":class:`numpy.ndarray`",
                                         ":class:`dask.array.Array`")])
