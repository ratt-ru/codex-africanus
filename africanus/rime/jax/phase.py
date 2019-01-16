# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp

try:
    import jax.numpy as np
except ImportError as e:
    opt_import_error = e
else:
    opt_import_error = None

from africanus.constants import minus_two_pi_over_c
import africanus.util.jax_init
from africanus.util.requirements import requires_optional


@requires_optional('jax', opt_import_error)
def phase_delay(lm, uvw, frequency):
    global minus_two_pi_over_c

    const_type = np.result_type(lm, uvw, frequency)
    out_dtype = np.result_type(const_type, np.complex64)

    one = const_type.type(1.0)
    complex_one = out_dtype.type(1j)
    minus_two_pi_over_c = const_type.type(minus_two_pi_over_c)

    l = lm[:, 0, None, None]
    m = lm[:, 1, None, None]

    u = uvw[None, :, 0, None]
    v = uvw[None, :, 1, None]
    w = uvw[None, :, 2, None]

    n = np.sqrt(one - l**2 - m**2) - one

    real_phase = (l * u + m * v + n * w)
    real_phase = minus_two_pi_over_c * real_phase
    real_phase *= frequency[None, None, :]

    return np.exp(complex_one*real_phase)
