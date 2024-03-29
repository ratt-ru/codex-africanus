# -*- coding: utf-8 -*-


try:
    import jax.numpy as jnp
except ImportError as e:
    opt_import_error = e
else:
    opt_import_error = None

from africanus.constants import minus_two_pi_over_c
from africanus.util.requirements import requires_optional


@requires_optional("jax", opt_import_error)
def phase_delay(lm, uvw, frequency):
    one = lm.dtype.type(1.0)
    neg_two_pi_over_c = lm.dtype.type(minus_two_pi_over_c)

    l = lm[:, 0, None, None]  # noqa
    m = lm[:, 1, None, None]

    u = uvw[None, :, 0, None]
    v = uvw[None, :, 1, None]
    w = uvw[None, :, 2, None]

    n = jnp.sqrt(one - l**2 - m**2) - one

    real_phase = neg_two_pi_over_c * (l * u + m * v + n * w) * frequency[None, None, :]

    return jnp.exp(jnp.complex64(1j) * real_phase)
