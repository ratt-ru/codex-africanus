# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .kernels import im_to_vis_kernel_docs, vis_to_im_kernel_docs
from .kernels import im_to_vis_kernel as np_im_to_vis_kernel
from .kernels import vis_to_im_kernel as np_vis_to_im_kernel

from ..util.docs import on_rtd, doc_tuple_to_str, mod_docs
from ..util.requirements import have_packages, MissingPackageException

_package_requirements = ('dask.array',)
have_requirements = have_packages(*_package_requirements)

if not have_requirements or on_rtd():
    def im_to_vis_kernel(uvw, lm, frequency, dtype=None):
        raise MissingPackageException(*_package_requirements)

    def vis_to_im_kernel(uvw, lm, frequency, dtype=None):
        raise MissingPackageException(*_package_requirements)
else:
    import numpy as np
    import dask.array as da

    def im_to_vis_kernel(uvw, lm, frequency, dtype=np.complex128):
        """ Dask wrapper for phase_delay function """
        def _wrapper(uvw, lm, frequency, dtype_):
            return np_im_to_vis_kernel(uvw[0], lm[0], frequency, dtype=dtype_)

        return da.core.atop(_wrapper, ("row", "source", "chan"),
                            uvw, ("row", "(u,v,w)"),
                            lm, ("source", "(l,m)"),
                            frequency, ("chan",),
                            dtype=dtype,
                            dtype_=dtype)

    def vis_to_im_kernel(uvw, lm, frequency, dtype=np.complex128):
        """ Dask wrapper for phase_delay_adjoint function """
        def _wrapper(uvw, lm, frequency, dtype_):
            return np_vis_to_im_kernel(uvw[0], lm[0], frequency, dtype=dtype_)

        return da.core.atop(_wrapper, ("source", "row", "chan"),
                            uvw, ("row", "(u,v,w)"),
                            lm, ("source", "(l,m)"),
                            frequency, ("chan",),
                            dtype=dtype,
                            dtype_=dtype)


im_to_vis_kernel.__doc__ = doc_tuple_to_str(im_to_vis_kernel_docs,
                                       [(":class:`numpy.ndarray`",
                                         ":class:`dask.array.Array`")])

vis_to_im_kernel.__doc__ = doc_tuple_to_str(vis_to_im_kernel_docs,
                                       [(":class:`numpy.ndarray`",
                                         ":class:`dask.array.Array`")])
