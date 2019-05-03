# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from africanus.model.shape.gaussian_shape import (gaussian as np_gaussian,
                                                  GAUSSIAN_DOCS)
from africanus.util.requirements import requires_optional

from africanus.model.shape.shapelets import shapelet as nb_shapelet

import numpy as np

try:
    import dask.array as da
except ImportError as e:
    opt_import_error = e
else:
    opt_import_error = None


def _wrapper(uvw, frequency, shape_params):
    return np_gaussian(uvw[0], frequency, shape_params[0])


@requires_optional('dask.array', opt_import_error)
def gaussian(uvw, frequency, shape_params):
    dtype = np.result_type(uvw.dtype, frequency.dtype, shape_params.dtype)

    return da.blockwise(_wrapper, ("source", "row", "chan"),
                        uvw, ("row", "uvw-comp"),
                        frequency, ("chan",),
                        shape_params, ("source", "shape-comp"),
                        dtype=dtype)

def _shapelet_wrapper(coords, frequency, coeffs, beta):
    return nb_shapelet(coords[0], frequency, coeffs[0][0], beta[0])

@requires_optional('dask.array', opt_import_error)
def shapelet(coords, frequency, coeffs, beta):
    dtype = np.complex128
    return da.blockwise(_shapelet_wrapper, ("source", "row", "chan" ),
                        coords, ("row", "uvw-comp"),
                        frequency, ("chan",),
                        coeffs, ("source", "nmax1", "nmax2"),
                        beta, ("source", "beta-comp"), dtype=dtype)

try:
    gaussian.__doc__ = GAUSSIAN_DOCS.substitute(
                            array_type=":class:`dask.array.Array`")
except AttributeError:
    pass
