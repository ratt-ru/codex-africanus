# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import wraps

from .component_spi import SPI_DOCSTRING
from .component_spi import fit_spi_components as np_fit_spi_components

from ...util.requirements import requires_optional

import numpy as np

try:
    import dask.array as da
except ImportError:
    pass


@wraps(np_fit_spi_components)
def _fit_spi_components_wrapper(data, weights, freqs, freq0,
                                alphai, I0i, tol_, maxiter_,
                                dtype_):
    return np_fit_spi_components(data[0],
                                 weights[0],
                                 freqs[0],
                                 freq0,
                                 alphai[0] if alphai is not None else alphai,
                                 I0i[0] if I0i is not None else I0i,
                                 tol=tol_,
                                 maxiter=maxiter_,
                                 dtype=dtype_)


@requires_optional('dask.array')
def fit_spi_components(data, weights, freqs, freq0,
                       alphai=None, I0i=None,
                       tol=1e-6, maxiter=100,
                       dtype=np.float64):
    """ Dask wrapper fit_spi_components function """
    return da.core.atop(_fit_spi_components_wrapper, ("vars", "comps"),
                        data, ("comps", "chan"),
                        weights, ("chan",),
                        freqs, ("chan",),
                        freq0, None,
                        alphai, ("comps",) if alphai is not None else None,
                        I0i, ("comps",) if I0i is not None else None,
                        tol, None,
                        maxiter, None,
                        dtype, None,
                        new_axes={"vars": 4},
                        dtype=dtype)


fit_spi_components.__doc__ = SPI_DOCSTRING.substitute(
                        array_type=":class:`dask.array.Array`")
