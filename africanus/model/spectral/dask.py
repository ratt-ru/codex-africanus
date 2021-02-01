# -*- coding: utf-8 -*-


from africanus.model.spectral.spec_model import (
                                        spectral_model as np_spectral_model,
                                        SPECTRAL_MODEL_DOC)
from africanus.util.requirements import requires_optional

try:
    import dask.array as da
except ImportError as e:
    opt_import_error = e
else:
    opt_import_error = None


def _wrapper(stokes, spi, ref_freq, frequencies, base=None):
    return np_spectral_model(stokes, spi[0], ref_freq, frequencies, base=base)


@requires_optional("dask.array", opt_import_error)
def spectral_model(stokes, spi, ref_freq, frequencies, base=0):
    if len(spi.chunks[1]) != 1:
        raise ValueError("Chunking along the spi dimension unsupported")

    pol_dim = () if stokes.ndim == 1 else ("pol",)

    return da.blockwise(_wrapper, ("source", "chan",) + pol_dim,
                        stokes, ("source",) + pol_dim,
                        spi, ("source", "spi") + pol_dim,
                        ref_freq, ("source",),
                        frequencies, ("chan",),
                        base=base,
                        dtype=stokes.dtype)


try:
    spectral_model.__doc__ = SPECTRAL_MODEL_DOC.substitute(
                                array_type=":class:`dask.array.Array`")
except AttributeError:
    pass
