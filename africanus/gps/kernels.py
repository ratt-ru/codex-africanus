# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from africanus.gps.utils import abs_diff

def exponential_squared(x, xp, sigmaf, l):
    """
    Create exponential squared covariance function between inputs x and xp
    """
    xxp = abs_diff(x, xp)
    return sigmaf**2*np.exp(-xxp**2/(2*l**2))