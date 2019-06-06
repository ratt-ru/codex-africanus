# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import wraps
import math

import numpy as np

from africanus.constants import minus_two_pi_over_c
from africanus.util.docs import DocstringTemplate
from africanus.util.numba import generated_jit, njit, is_numba_type_none
from africanus.util.type_inference import infer_complex_dtype


def _out_factory(out_present):
    if out_present:
        def impl(out, shape, dtype):
            # TODO(sjperkins) Check the dtype too?
            if out.shape != shape:
                raise ValueError("out.shape does not match expected shape")

            return out
    else:
        def impl(out, shape, dtype):
            return np.zeros(shape, dtype)

    return njit(nogil=True, cache=True)(impl)


@generated_jit(nopython=True, nogil=True, cache=True)
def phase_delay(lm, uvw, frequency, out=None):
    have_out = not is_numba_type_none(out)

    # Bake constants in with the correct type
    one = lm.dtype(1.0)
    neg_two_pi_over_c = lm.dtype(minus_two_pi_over_c)
    out_dtype = infer_complex_dtype(lm, uvw, frequency)

    create_output = _out_factory(have_out)

    @wraps(phase_delay)
    def _phase_delay_impl(lm, uvw, frequency, out=None):
        shape = (lm.shape[0], uvw.shape[0], frequency.shape[0])
        complex_phase = create_output(out, shape, out_dtype)

        # For each source
        for source in range(lm.shape[0]):
            l, m = lm[source]
            n = math.sqrt(one - l**2 - m**2) - one

            # For each uvw coordinate
            for row in range(uvw.shape[0]):
                u, v, w = uvw[row]
                # e^(-2*pi*(l*u + m*v + n*w)/c)
                real_phase = neg_two_pi_over_c * (l * u + m * v + n * w)

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


PHASE_DELAY_DOCS = DocstringTemplate(
    r"""
    Computes the phase delay (K) term:

    .. math::

        & {\Large e^{-2 \pi i (u l + v m + w (n - 1))} }

        & \textrm{where } n = \sqrt{1 - l^2 - m^2}

    Notes
    -----

    Corresponds to the complex exponential of the `Van Cittert-Zernike Theorem
    <https://en.wikipedia.org/wiki/Van_Cittert%E2%80%93Zernike_theorem_>`_.

    `MeqTrees
    <https://github.com/ska-sa/meqtrees-timba/blob/
    6a7e873d4d1fe538981dec5851418cbd371b8388/MeqNodes/src/PSVTensor.cc#L314_>`_
    uses a positive sign convention and so any UVW coordinates must be inverted
    in order for their phase delay terms (and therefore visibilities) to agree.

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
    out : $(array_type), optional
        Array holding the output results. Should have the
        same shape as the returned `complex_phase`.

    Returns
    -------
    complex_phase : $(array_type)
        complex of shape :code:`(source, row, chan)`
    """)

try:
    phase_delay.__doc__ = PHASE_DELAY_DOCS.substitute(
                            array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass
