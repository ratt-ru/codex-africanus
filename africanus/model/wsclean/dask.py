# -*- coding: utf-8 -*-


from africanus.model.wsclean.spec_model import (spectra as np_spectra,
                                                SPECTRA_DOCS)
from africanus.util.requirements import requires_optional

try:
    import dask.array as da
except ImportError as e:
    opt_import_error = e
else:
    opt_import_error = None


def spectra_wrapper(stokes, spi, log_si, ref_freq, frequency):
    return np_spectra(stokes, spi[0], log_si, ref_freq, frequency)


@requires_optional('dask.array', opt_import_error)
def spectra(stokes, spi, log_si, ref_freq, frequency):
    corrs = tuple("corr-%d" % i for i in range(len(stokes.shape[1:])))
    log_si_schema = None if isinstance(log_si, bool) else ("source",)

    return da.blockwise(spectra_wrapper, ("source", "chan") + corrs,
                        stokes, ("source",) + corrs,
                        spi, ("source", "spi") + corrs,
                        log_si, log_si_schema,
                        ref_freq, ("source",),
                        frequency, ("chan",),
                        dtype=stokes.dtype)


try:
    spectra.__doc__ = SPECTRA_DOCS.substitute(
                            array_type=":class:`dask.array.Array`")
except AttributeError:
    pass
