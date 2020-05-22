# -*- coding: utf-8 -*-


from numba import types
import numpy as np

from africanus.util.numba import generated_jit, njit
from africanus.util.docs import DocstringTemplate


def numpy_spectral_model(stokes, spi, ref_freq, frequency, base):
    out_shape = (stokes.shape[0], frequency.shape[0]) + stokes.shape[1:]

    # Add in missing pol dimensions
    if stokes.ndim == 1:
        stokes = stokes[:, None]

    if spi.ndim == 2:
        spi = spi[:, :, None]

    npol = spi.shape[2]

    if isinstance(base, list):
        base = base + [base[-1]] * (npol - len(base))
    else:
        base = [base]*npol

    spi_exps = np.arange(1, spi.shape[1] + 1)

    spectral_model = np.empty((stokes.shape[0], frequency.shape[0], npol),
                              dtype=stokes.dtype)

    spectral_model[:, :, :] = stokes[:, None, :]

    for p, b in enumerate(base):
        if b in ("std", 0):
            freq_ratio = (frequency[None, :] / ref_freq[:, None])
            term = freq_ratio[:, None, :]**spi[:, :, p, None]
            spectral_model[:, :, p] *= term.prod(axis=1)
        elif b in ("log", 1):
            freq_ratio = np.log(frequency[None, :] / ref_freq[:, None])
            term = freq_ratio[:, None, :]**spi_exps[None, :, None]
            term = spi[:, :, p, None] * term
            spectral_model[:, :, p] = (np.exp(np.log(stokes[:, p, None]) +
                                              term.sum(axis=1)))
        elif b in ("log10", 2):
            freq_ratio = np.log10(frequency[None, :] / ref_freq[:, None])
            term = freq_ratio[:, None, :]**spi_exps[None, :, None]
            term = spi[:, :, p, None] * term
            spectral_model[:, :, p] = (10**(np.log10(stokes[:, p, None]) +
                                            term.sum(axis=1)))
        else:
            raise ValueError("Invalid base %s" % base)

    return spectral_model.reshape(out_shape)


def pol_getter_factory(npoldims):
    if npoldims == 0:
        def impl(pol_shape):
            return 1
    else:
        def impl(pol_shape):
            npols = 1

            for c in pol_shape:
                npols *= c

            return npols

    return njit(nogil=True, cache=True)(impl)


def promote_base_factory(is_base_list):
    if is_base_list:
        def impl(base, npol):
            return base + [base[-1]] * (npol - len(base))
    else:
        def impl(base, npol):
            return [base] * npol

    return njit(nogil=True, cache=True)(impl)


def add_pol_dim_factory(have_pol_dim):
    if have_pol_dim:
        def impl(array):
            return array
    else:
        def impl(array):
            return array.reshape(array.shape + (1,))

    return njit(nogil=True, cache=True)(impl)


@generated_jit(nopython=True, nogil=True, cache=True)
def spectral_model(stokes, spi, ref_freq, frequency, base=0):
    arg_dtypes = tuple(np.dtype(a.dtype.name) for a
                       in (stokes, spi, ref_freq, frequency))
    dtype = np.result_type(*arg_dtypes)

    if isinstance(base, types.containers.List):
        is_base_list = True
        base = base.dtype
    else:
        is_base_list = False

    promote_base = promote_base_factory(is_base_list)

    if isinstance(base, types.scalars.Integer):
        def is_std(base):
            return base == 0

        def is_log(base):
            return base == 1

        def is_log10(base):
            return base == 2

    elif isinstance(base, types.misc.UnicodeType):
        def is_std(base):
            return base == "std"

        def is_log(base):
            return base == "log"

        def is_log10(base):
            return base == "log10"
    else:
        raise TypeError("base '%s' should be a string or integer" % base)

    is_std = njit(nogil=True, cache=True)(is_std)
    is_log = njit(nogil=True, cache=True)(is_log)
    is_log10 = njit(nogil=True, cache=True)(is_log10)

    npoldims = stokes.ndim - 1
    pol_get_fn = pol_getter_factory(npoldims)
    add_pol_dim = add_pol_dim_factory(npoldims > 0)

    if spi.ndim - 2 != npoldims:
        raise ValueError("Dimensions on stokes and spi don't agree")

    def impl(stokes, spi, ref_freq, frequency, base=0):
        nsrc = stokes.shape[0]
        nchan = frequency.shape[0]
        nspi = spi.shape[1]
        npol = pol_get_fn(stokes.shape[1:])

        if npol != pol_get_fn(spi.shape[2:]):
            raise ValueError("Correlations on stokes and spi don't agree")

        # Promote base argument to a per-polarisation list
        list_base = promote_base(base, npol)

        # Reshape adding a polarisation dimension if necessary
        estokes = add_pol_dim(stokes)
        espi = add_pol_dim(spi)

        spectral_model = np.empty((nsrc, nchan, npol), dtype=dtype)

        # TODO(sjperkins)
        # Polarisation + associated base on the outer loop
        # The output cache patterns could be improved.
        for p, b in enumerate(list_base[:npol]):
            if is_std(b):
                for s in range(nsrc):
                    rf = ref_freq[s]

                    for f in range(nchan):
                        freq_ratio = frequency[f] / rf
                        spec_model = estokes[s, p]

                        for si in range(0, nspi):
                            term = freq_ratio ** espi[s, si, p]
                            spec_model *= term

                        spectral_model[s, f, p] = spec_model

            elif is_log(b):
                for s in range(nsrc):
                    rf = ref_freq[s]

                    for f in range(nchan):
                        freq_ratio = np.log(frequency[f] / rf)
                        spec_model = np.log(estokes[s, p])

                        for si in range(0, nspi):
                            term = espi[s, si, p] * freq_ratio**(si + 1)
                            spec_model += term

                        spectral_model[s, f, p] = np.exp(spec_model)

            elif is_log10(b):
                for s in range(nsrc):
                    rf = ref_freq[s]

                    for f in range(nchan):
                        freq_ratio = np.log10(frequency[f] / rf)
                        spec_model = np.log10(estokes[s, p])

                        for si in range(0, nspi):
                            term = espi[s, si, p] * freq_ratio**(si + 1)
                            spec_model += term

                        spectral_model[s, f, p] = 10**spec_model

            else:
                raise ValueError("Invalid base")

        out_shape = (stokes.shape[0], frequency.shape[0]) + stokes.shape[1:]
        return spectral_model.reshape(out_shape)

    return impl


SPECTRAL_MODEL_DOC = DocstringTemplate(r"""
Compute a spectral model, per polarisation.

.. math::
    :nowrap:

    \begin{eqnarray}
    I(\lambda) & = & I_0 \prod_{i=1} (\lambda / \lambda_0)^{\alpha_{i}} \\
    \ln( I(\lambda) ) & = & \sum_{i=0} \alpha_{i}
      \ln (\lambda / \lambda_0)^i
      \, \textrm{where} \, \alpha_0 = \ln I_0 \\
    \log_{10}( I(\lambda) ) & = & \sum_{i=0} \alpha_{i}
      \log_{10} (\lambda / \lambda_0)^i
      \, \textrm{where} \, \alpha_0 = \log_{10} I_0 \\
    \end{eqnarray}

Parameters
----------
stokes : $(array_type)
    Stokes parameters of shape :code:`(source,)` or :code:`(source, pol)`.
    If a ``pol`` dimension is present, then it must also be present on ``spi``.
spi : $(array_type)
    Spectral index of shape :code:`(source, spi-comps)`
    or :code:`(source, spi-comps, pol)`.
ref_freq : $(array_type)
    Reference frequencies of shape :code:`(source,)`
frequencies : $(array_type)
    Frequencies of shape :code:`(chan,)`
base : {"std", "log", "log10"} or {0, 1, 2} or list.
    string or corresponding enumeration specifying the polynomial base.
    Defaults to 0.

    If a list is provided, a polynomial base can be specified for each
    stokes parameter or polarisation in the ``pol`` dimension.

    string specification of the base is only supported in python 3.
    while the corresponding integer enumerations are supported
    on all python versions.

Returns
-------
spectral_model : $(array_type)
    Spectral Model of shape :code:`(source, chan)` or
    :code:`(source, chan, pol)`.
""")

try:
    spectral_model.__doc__ = SPECTRAL_MODEL_DOC.substitute(
                                array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass
