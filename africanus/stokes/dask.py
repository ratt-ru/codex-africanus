# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .brightness import brightness as np_brightness

from ..util.docs import on_rtd, mod_docs
from ..util.requirements import have_packages, MissingPackageException

_package_requirements = ('dask.array',)
have_requirements = have_packages(*_package_requirements)


if not have_requirements or on_rtd():
    def brightness(stokes, polarisation_type=None):
        raise MissingPackageException(_*_package_requirements)
else:
    import dask.array as da
    import numpy as np

    def brightness(stokes, polarisation_type=None):
        if not stokes.shape[1] == stokes.chunks[1][0]:
            raise ValueError("The polarisation dimension "
                             "of the 'stokes' array "
                             "may not be chunked "
                             "(the chunk size must match "
                             "the dimension size).")

        return da.core.atop(np_brightness, ("stokes", "pol"),
                            stokes, ("stokes", "pol"),
                            polarisation_type=polarisation_type,
                            dtype=np.complex64 if stokes.dtype == np.float32
                            else np.complex128)


brightness.__doc__ = mod_docs(np_brightness.__doc__,
                              [(":class:`numpy.ndarray`",
                                ":class:`dask.array.Array`")])
