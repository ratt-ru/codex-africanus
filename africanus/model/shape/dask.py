# -*- coding: utf-8 -*-


from africanus.model.shape.gaussian_shape import (gaussian as np_gaussian,
                                                  GAUSSIAN_DOCS)
from africanus.util.requirements import requires_optional

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


try:
    gaussian.__doc__ = GAUSSIAN_DOCS.substitute(
                            array_type=":class:`dask.array.Array`")
except AttributeError:
    pass
