# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .phase import phase_delay_docs
from ..util.docs import on_rtd, doc_tuple_to_str
from ..util.requirements import have_packages, MissingPackageException

_package_requirements = ('dask.array',)
have_requirements = have_packages(*_package_requirements)

if not have_requirements or on_rtd():
    def phase_delay(uvw, lm, frequency, dtype=None):
        raise MissingPackageException(*_package_requirements)
else:
    import numpy as np
    import dask.array as da
    from .phase import phase_delay as _np_phase_delay

    def phase_delay(uvw, lm, frequency, dtype=np.complex128):
        """ Dask wrapper for phase_delay function """
        def _wrapper(uvw, lm, frequency, dtype_):
            return _np_phase_delay(uvw[0], lm[0], frequency, dtype=dtype_)

        return da.core.atop(_wrapper, ("source", "row", "chan"),
                            uvw, ("row", "(u,v,w)"),
                            lm, ("source", "(l,m)"),
                            frequency, ("chan",),
                            dtype=dtype,
                            dtype_=dtype)

phase_delay.__doc__ = doc_tuple_to_str(phase_delay_docs, replacements=[
    (":class:`numpy.ndarray`", ":class:`dask.array.Array`")])
