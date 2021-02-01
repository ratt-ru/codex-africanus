# -*- coding: utf-8 -*-


from africanus.model.spi.component_spi import SPI_DOCSTRING
from africanus.model.spi.component_spi import (
                                fit_spi_components as np_fit_spi_components)

from africanus.util.requirements import requires_optional

try:
    from dask.array.core import blockwise
except ImportError as e:
    opt_import_error = e
else:
    opt_import_error = None


def _fit_spi_components_wrapper(data, weights, freqs, freq0,
                                alphai, I0i, beam, tol, maxiter):
    return np_fit_spi_components(data[0],
                                 weights[0],
                                 freqs[0],
                                 freq0,
                                 alphai,
                                 I0i,
                                 beam[0] if beam is not None else beam,
                                 tol=tol,
                                 maxiter=maxiter)


@requires_optional('dask.array', opt_import_error)
def fit_spi_components(data, weights, freqs, freq0,
                       alphai=None, I0i=None, beam=None,
                       tol=1e-5, maxiter=100):
    """ Dask wrapper fit_spi_components function """
    return blockwise(_fit_spi_components_wrapper, ("vars", "comps"),
                     data, ("comps", "chan"),
                     weights, ("chan",),
                     freqs, ("chan",),
                     freq0, None,
                     alphai, ("comps",) if alphai is not None else None,
                     I0i, ("comps",) if I0i is not None else None,
                     beam, ("comps", "chan") if beam is not None else None,
                     tol, None,
                     maxiter, None,
                     new_axes={"vars": 4},
                     dtype=data.dtype)


fit_spi_components.__doc__ = SPI_DOCSTRING.substitute(
                        array_type=":class:`dask.array.Array`")
