# -*- coding: utf-8 -*-


from africanus.model.shape.gaussian_shape import (
    gaussian as np_gaussian,
    GAUSSIAN_DOCS,
)
from africanus.util.requirements import requires_optional

from africanus.model.shape.shapelets import shapelet as nb_shapelet
from africanus.model.shape.shapelets import (
    shapelet_with_w_term as nb_shapelet_with_w_term,
)

import numpy as np

try:
    import dask.array as da
except ImportError as e:
    opt_import_error = e
else:
    opt_import_error = None


def gaussian_wrapper(uvw, frequency, shape_params):
    return np_gaussian(uvw[0], frequency, shape_params[0])


@requires_optional("dask.array", opt_import_error)
def gaussian(uvw, frequency, shape_params):
    dtype = np.result_type(uvw.dtype, frequency.dtype, shape_params.dtype)

    return da.blockwise(
        gaussian_wrapper,
        ("source", "row", "chan"),
        uvw,
        ("row", "uvw-comp"),
        frequency,
        ("chan",),
        shape_params,
        ("source", "shape-comp"),
        dtype=dtype,
    )


def _shapelet_wrapper(coords, frequency, coeffs, beta, delta_lm):
    return nb_shapelet(
        coords[0], frequency, coeffs[0][0], beta[0], delta_lm[0]
    )


@requires_optional("dask.array", opt_import_error)
def shapelet(coords, frequency, coeffs, beta, delta_lm):
    dtype = np.complex128
    return da.blockwise(
        _shapelet_wrapper,
        ("row", "chan", "source"),
        coords,
        ("row", "coord-comp"),
        frequency,
        ("chan",),
        coeffs,
        ("source", "nmax1", "nmax2"),
        beta,
        ("source", "beta-comp"),
        delta_lm,
        ("delta_lm-comp",),
        dtype=dtype,
    )


def _shapelet_with_w_term_wrapper(
    coords, frequency, coeffs, beta, delta_lm, lm
):
    return nb_shapelet_with_w_term(
        coords[0], frequency, coeffs[0][0], beta[0], delta_lm[0], lm[0]
    )


@requires_optional("dask.array", opt_import_error)
def shapelet_with_w_term(coords, frequency, coeffs, beta, delta_lm, lm):
    dtype = np.complex128
    return da.blockwise(
        _shapelet_with_w_term_wrapper,
        ("row", "chan", "source"),
        coords,
        ("row", "coord-comp"),
        frequency,
        ("chan",),
        coeffs,
        ("source", "nmax1", "nmax2"),
        beta,
        ("source", "beta-comp"),
        delta_lm,
        ("delta_lm-comp",),
        lm,
        ("source", "lm-comp"),
        dtype=dtype,
    )


try:
    gaussian.__doc__ = GAUSSIAN_DOCS.substitute(
        array_type=":class:`dask.array.Array`"
    )
except AttributeError:
    pass
