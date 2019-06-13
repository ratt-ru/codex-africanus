__all__ = ["c", "minus_two_pi_over_c", "two_pi_over_c",
           "arcseconds_to_radians"]

import numpy as np

# Lightspeed
c = 2.99792458e8

two_pi_over_c = 2 * np.pi / c
minus_two_pi_over_c = -two_pi_over_c
arcseconds_to_radians = np.deg2rad(1.0/(60.0*60.0))
