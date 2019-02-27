# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numba


@numba.njit(nogil=True)
def row_iterator(rows, bin_size):
    row_bin = 0
    bin_contents = 0

    for row in range(rows - 1):
        # We're adding something to the bin
        # so we can reason about bin fullness upfront
        bin_contents += 1
        bin_full = bin_contents == bin_size

        # Yield row, bin and whether the bin is full
        yield row, row_bin, bin_full

        # Reset bin variables if the bin is full
        if bin_full:
            row_bin += 1
            bin_contents = 0

    # Bin is always full on the last iteration
    yield row + 1, row_bin, True
