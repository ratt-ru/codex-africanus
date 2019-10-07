# -*- coding: utf-8 -*-


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

    one = lm.dtype.type(1.0)
    neg_two_pi_over_c = lm.dtype.type(minus_two_pi_over_c)
    complex_one = out_dtype.type(1j)

    l = lm[:, 0, None, None]  # noqa
    m = lm[:, 1, None, None]

    u = uvw[None, :, 0, None]
    v = uvw[None, :, 1, None]
    w = uvw[None, :, 2, None]

    n = np.sqrt(one - l**2 - m**2) - one

    real_phase = (neg_two_pi_over_c *
                  (l * u + m * v + n * w) *
                  frequency[None, None, :])

    return np.exp(complex_one*real_phase)
