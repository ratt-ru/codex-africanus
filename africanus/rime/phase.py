# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import math

import numba
import numpy as np

from ..constants import c as lightspeed, minus_two_pi_over_c
from ..util.docs import doc_tuple_to_str


def phase_delay(uvw, lm, frequency, dtype=None):

    @numba.jit(nopython=True, nogil=True, cache=True)
    def _phase_delay_impl(uvw, lm, frequency, complex_phase):
        # For each source
        for source in range(lm.shape[0]):
            l, m = lm[source]
            n = math.sqrt(1.0 - l**2 - m**2) - 1.0

            # For each uvw coordinate
            for row in range(uvw.shape[0]):
                u, v, w = uvw[row]
                # e^(-2*pi*(l*u + m*v + n*w)/c)
                real_phase = minus_two_pi_over_c * (l * u + m * v + n * w)

                # Multiple in frequency for each channel
                for chan in range(frequency.shape[0]):
                    p = real_phase * frequency[chan]

                    # Our phase input is purely imaginary
                    # so we can can elide a call to exp
                    # and just compute the cos and sin
                    complex_phase.real[source, row, chan] = math.cos(p)
                    complex_phase.imag[source, row, chan] = math.sin(p)

        return complex_phase

    complex_phase = np.empty((lm.shape[0], uvw.shape[0], frequency.shape[0]),
                             dtype=np.complex128 if dtype is None else dtype)

    return _phase_delay_impl(uvw, lm, frequency, complex_phase)


_DFT_DOCSTRING = namedtuple(
    "_DFTDOCSTRING", ["preamble", "parameters", "returns"])

phase_delay_docs = _DFT_DOCSTRING(
    preamble="""
    Computes the phase delay (K) term:

    .. math::

        & {\Large e^{-2 \pi i (u l + v m + w n)} }

        & \\textrm{where } n = \sqrt{1 - l^2 - m^2} - 1
    """,

    parameters="""
    Parameters
    ----------

    uvw : :class:`numpy.ndarray`
        UVW coordinates of shape :code:`(row, 3)` with
        U, V and W components in the last dimension.
    lm : :class:`numpy.ndarray`
        LM coordinates of shape :code:`(source, 2)` with
        L and M components in the last dimension.
    frequency : :class:`numpy.ndarray`
        frequencies of shape :code:`(chan,)`
    dtype : np.dtype, optional
        Datatype of result. Should be either np.complex64
        or np.complex128. Defaults to np.complex128
    """,

    returns="""
    Returns
    -------
    :class:`numpy.ndarray`
        complex of shape :code:`(source, row, chan)`
    """
)


phase_delay.__doc__ = doc_tuple_to_str(phase_delay_docs)
