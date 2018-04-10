# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .brightness import brightness as np_brightness, bright_corr_shape

from ..util.docs import on_rtd, mod_docs
from ..util.requirements import have_packages, MissingPackageException

_package_requirements = ('dask.array',)
have_requirements = have_packages(*_package_requirements)


if not have_requirements or on_rtd():
    def brightness(stokes, polarisation_type=None, corr_shape=None):
        raise MissingPackageException(_*_package_requirements)
else:
    import dask.array as da
    import numpy as np

    def brightness(stokes, polarisation_type=None, corr_shape=None):
        if corr_shape is None:
            corr_shape = 'flat'

        head, npol = stokes.shape[:-1], stokes.shape[-1]

        if not npol == stokes.chunks[-1][0]:
            raise ValueError("The polarisation dimension "
                             "of the 'stokes' array "
                             "may not be chunked "
                             "(the chunk size must match "
                             "the dimension size).")

        head_dims = tuple("head-%d" % i for i in range(len(head)))
        corr_shapes = bright_corr_shape(npol, corr_shape)
        corr_dims = tuple("corr-%d" % i for i in range(len(corr_shapes)))
        new_axes = {d: s for d, s in zip(corr_dims, corr_shapes)}

        def _wrapper(stokes):
            return np_brightness(stokes[0],
                                polarisation_type=polarisation_type,
                                corr_shape=corr_shape)

        return da.core.atop(_wrapper, head_dims + corr_dims,
                            stokes, head_dims + ("pol",),
                            new_axes=new_axes,
                            dtype=np.complex64 if stokes.dtype == np.float32
                            else np.complex128)

brightness.__doc__ = mod_docs(np_brightness.__doc__,
                              [(":class:`numpy.ndarray`",
                                ":class:`dask.array.Array`")])
