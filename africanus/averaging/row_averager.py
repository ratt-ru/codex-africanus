# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numba

from africanus.util.numba import is_numba_type_none


def output_factory(present):
    if present:
        def impl(new_shape, in_array):
            return np.zeros(new_shape, in_array.dtype)
    else:
        def impl(new_shape, in_array):
            return None

    return numba.njit(nogil=True, cache=True)(impl)


def add_factory(present):
    if present:
        def impl(output, orow, input, irow):
            output[orow] += input[irow]
    else:
        def impl(input, irow, output, orow):
            pass

    return numba.njit(nogil=True, cache=True)(impl)


@numba.generated_jit(nopython=True, nogil=True, cache=True)
def row_average(time, ant1, ant2, metadata,
                interval=None, exposure=None):
    have_interval = not is_numba_type_none(interval)
    have_exposure = not is_numba_type_none(exposure)

    interval_factory = output_factory(have_interval)
    exposure_factory = output_factory(have_exposure)

    interval_adder = add_factory(have_interval)
    exposure_adder = add_factory(have_exposure)

    print(have_interval, have_exposure)

    def impl(time, ant1, ant2, metadata,
             interval=None, exposure=None):
        (in_lookup, time_lookup, out_lookup,
         out_rows, time_bin_size, sentinel) = metadata

        nbl = in_lookup.shape[0]
        ntime = in_lookup.shape[1]

        time_bins = (ntime + time_bin_size - 1) // time_bin_size

        time_avg = np.empty(out_rows, time.dtype)
        ant1_avg = np.empty(out_rows, ant1.dtype)
        ant2_avg = np.empty(out_rows, ant2.dtype)
        interval_avg = interval_factory(out_rows, interval)
        exposure_avg = exposure_factory(out_rows, exposure)

        for bli in range(nbl):
            off = bli*time_bins
            tbin = numba.uint32(0)
            nbin_values = numba.uint32(0)

            for ti in range(ntime):
                # Lookup input row for this baseline and time
                in_row = in_lookup[bli, ti]

                if in_row == -1:
                    continue

                # Lookup output row
                out_row = out_lookup[off + tbin]

                nbin_values += 1

                # Here we can simply assign
                time_avg[out_row] = time_lookup[bli, tbin]
                ant1_avg[out_row] = ant1[in_row]
                ant2_avg[out_row] = ant2[in_row]

                interval_adder(interval_avg, out_row, interval, in_row)
                exposure_adder(exposure_avg, out_row, exposure, in_row)

                if nbin_values == time_bin_size:
                    tbin += 1
                    nbin_values = numba.uint32(0)

            if nbin_values > 0:
                tbin += 1

        return time_avg, ant1_avg, ant2_avg, interval_avg, exposure_avg

    return impl
