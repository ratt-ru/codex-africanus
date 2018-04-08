#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .transform import transform_sources as np_transform_sources

from ..util.docs import on_rtd, doc_tuple_to_str
from ..util.requirements import have_packages, MissingPackageException

_package_requirements = ('dask.array',)
have_requirements = have_packages(*_package_requirements)

if not have_requirements or on_rtd():
    def transform_sources(lm, parallactic_angles, pointing_errors,
                                    antenna_scaling, dtype=None):
        raise MissingPackageException(*_package_requirements)
else:
    import numpy as np
    import dask.array as da

    def transform_sources(lm, parallactic_angles, pointing_errors,
                            antenna_scaling, dtype=None):

        def _wrapper(lm, parallactic_angles, pointing_errors,
                        antenna_scaling, dtype_):
            return np_transform_sources(lm, parallactic_angles,
                    pointing_errors, antenna_scaling, dtype=dtype_)

        if dtype is None:
            dtype = np.float64

        return da.core.atop(_wrapper, ("src", "time", "ant", "chan", 'lm'),
                            lm, ("src", "lm"),
                            parallactic_angles, ("time", "ant"),
                            pointing_errors, ("time", "ant", "lm"),
                            antenna_scaling, ("ant", "chan"),
                            dtype=dtype,
                            dtype_=dtype)

