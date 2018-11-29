# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
from functools import wraps
import math

import numba
import numpy as np

from ..constants import minus_two_pi_over_c
from ..util.docs import DocstringTemplate, on_rtd
from ..util.numba import is_numba_type_none
from ..util.type_inference import infer_complex_dtype


def phase_delay(lm, uvw, frequency):
    out_dtype = infer_complex_dtype(lm, uvw, frequency)

    @wraps(phase_delay)
    def _phase_delay_impl(lm, uvw, frequency):
        shape = (lm.shape[0], uvw.shape[0], frequency.shape[0])
        complex_phase = np.zeros(shape, dtype=out_dtype)

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

    return _phase_delay_impl


if not on_rtd():
    jitter = numba.generated_jit(nopython=True, nogil=True, cache=True)
    phase_delay = jitter(phase_delay)


PHASE_DELAY_DOCS = DocstringTemplate(
    r"""
    Computes the phase delay (K) term:

    .. math::

        & {\Large e^{-2 \pi i (u l + v m + w n)} }

        & \textrm{where } n = \sqrt{1 - l^2 - m^2} - 1

    Parameters
    ----------

    lm : $(array_type)
        LM coordinates of shape :code:`(source, 2)` with
        L and M components in the last dimension.
    uvw : $(array_type)
        UVW coordinates of shape :code:`(row, 3)` with
        U, V and W components in the last dimension.
    frequency : $(array_type)
        frequencies of shape :code:`(chan,)`

    Returns
    -------
    $(array_type)
        complex of shape :code:`(source, row, chan)`
    """)

try:
    phase_delay.__doc__ = PHASE_DELAY_DOCS.substitute(
                            array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass
