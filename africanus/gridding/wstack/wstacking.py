from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def w_stacking_layers(w_min, w_max, l, m):
    """
    Computes the number of w-layers.

    Parameters
    ----------
    w_min : float
        Minimum W coordinate
    w_max : float
        Maximum W coordinate
    l : :class:`numpy.ndarray`
        l coordinates
    m : :class:`numpy.ndarray`
        m coordinates

    Returns
    -------
    int
        Number of w-layers
    """
    lm_max = np.sqrt(1 - l[None, :]**2 - m[:, None]**2).max()

    return np.ceil(2*np.pi*(w_max - w_min)/lm_max).astype(np.int32).item()
