__all__ = ["c", "minus_two_pi_over_c", "two_pi_over_c"]

import numpy as np

from ..util.docs import on_rtd

# Lightspeed
c = 2.99792458e8

two_pi_over_c = 2 * 3.14 / c if on_rtd() else 2 * np.pi / c
minus_two_pi_over_c = -two_pi_over_c
