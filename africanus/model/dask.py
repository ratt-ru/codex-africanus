# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import wraps

from africanus.model.spec_model import (spectral_model as np_spectral_model,
                                        SPECTRAL_MODEL_DOC)
from africanus.util.requirements import requires_optional

try:
    import dask.array as da
except ImportError as e:
    opt_import_error = e
else:
    opt_import_error = None


@wraps(np_spectral_model)
def _wrapper(stokes, spi, ref_freq, frequencies, base=None):
    return np_spectral_model(stokes, spi[0], ref_freq, frequencies, base=base)


@requires_optional("dask.array", opt_import_error)
def spectral_model(stokes, spi, ref_freq, frequencies, base):
    if len(spi.chunks[1]) != 1:
        raise ValueError("Chunking along the spi dimension unsupported")

    return da.blockwise(_wrapper, ("source", "chan", "corr"),
                        stokes, ("source", "corr"),
                        spi, ("source", "spi", "corr"),
                        ref_freq, ("source",),
                        frequencies, ("chan",),
                        base=base,
                        dtype=stokes.dtype)


try:
    spectral_model.__doc__ = SPECTRAL_MODEL_DOC.substitute(
                                array_type=":class:`dask.array.Array`")
except AttributeError:
    pass
