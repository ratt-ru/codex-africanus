# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numba

from africanus.util.numba import is_numba_type_none


@numba.jit(nopython=True, nogil=True, cache=True)
def row_average(time, ant1, ant2, metadata):
    (in_lookup, time_lookup, out_lookup,
     out_rows, time_bin_size, sentinel) = metadata

    nbl = in_lookup.shape[0]
    ntime = in_lookup.shape[1]

    time_bins = (ntime + time_bin_size - 1) // time_bin_size

    time_avg = np.full(out_rows, -1.0, dtype=time.dtype)
    ant1_avg = np.full(out_rows, -1, dtype=ant1.dtype)
    ant2_avg = np.full(out_rows, -1, dtype=ant2.dtype)

    for bli in range(nbl):
        off = bli*time_bins
        tbin = numba.uint32(0)
        nbin_values = numba.uint32(0)

        for ti in range(ntime):
            # Lookup input row for this baseline and time
            in_row = in_lookup[bli, ti]

            if in_row == -1 or time_lookup[bli, tbin] == sentinel:
                continue

            # Lookup output row
            out_row = out_lookup[off + tbin]

            nbin_values += 1

            time_avg[out_row] = time_lookup[bli, tbin]
            ant1_avg[out_row] = ant1[in_row]
            ant2_avg[out_row] = ant2[in_row]

            if nbin_values == time_bin_size:
                tbin += 1
                nbin_values = numba.uint32(0)

    return time_avg, ant1_avg, ant2_avg
