# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import jax.numpy as np
except ImportError as e:
    opt_import_error = e
else:
    opt_import_error = None

from africanus.constants import minus_two_pi_over_c
from africanus.util.requirements import requires_optional


@requires_optional('jax', opt_import_error)
def phase_delay(lm, uvw, frequency):
    out_dtype = np.result_type(lm, uvw, frequency, np.complex64)

    l = lm[:, 0, None, None]
    m = lm[:, 1, None, None]

    u = uvw[None, :, 0, None]
    v = uvw[None, :, 1, None]
    w = uvw[None, :, 2, None]

    n = np.sqrt(1.0 - l**2 - m**2) - 1.0

    real_phase = minus_two_pi_over_c * (l * u + m * v + n * w)
    real_phase *= frequency[None, None, :]

    return np.exp(1j*real_phase)
