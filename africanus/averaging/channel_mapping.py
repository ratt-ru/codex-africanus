# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numba


@numba.jit(nopython=True, nogil=True, cache=True)
def channel_mapper(nchan, chan_bin_size=1):
    chan_bins = (nchan + chan_bin_size - 1) // chan_bin_size

    chan_map = np.empty(nchan, dtype=np.uint32)

    chan_bin = 0
    bin_count = 0

    for c in range(nchan):
        chan_map[c] = chan_bin
        bin_count += 1

        if bin_count == chan_bin_size:
            chan_bin += 1
            bin_count = 0

    return chan_map
