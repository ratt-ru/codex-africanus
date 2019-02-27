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


def normaliser_factory(present):
    if present:
        def impl(data, idx, bin_size):
            data[idx] /= bin_size
    else:
        def impl(data, idx, bin_size):
            pass

    return numba.njit(nogil=True, cache=True)(impl)


@numba.generated_jit(nopython=True, nogil=True, cache=True)
def row_average(time, ant1, ant2, metadata,
                time_centroid=None,
                interval=None, exposure=None):
    have_time_centroid = not is_numba_type_none(time_centroid)
    have_interval = not is_numba_type_none(interval)
    have_exposure = not is_numba_type_none(exposure)

    centroid_factory = output_factory(time_centroid)
    interval_factory = output_factory(have_interval)
    exposure_factory = output_factory(have_exposure)

    centroid_adder = add_factory(have_time_centroid)
    interval_adder = add_factory(have_interval)
    exposure_adder = add_factory(have_exposure)

    centroid_normaliser = normaliser_factory(have_time_centroid)

    def impl(time, ant1, ant2, metadata,
             time_centroid=None,
             interval=None,
             exposure=None):
        (in_lookup, time_lookup, out_lookup,
         out_rows, time_bin_size, sentinel) = metadata

        nbl = in_lookup.shape[0]
        ntime = in_lookup.shape[1]

        time_bins = (ntime + time_bin_size - 1) // time_bin_size

        time_avg = np.empty(out_rows, time.dtype)
        ant1_avg = np.empty(out_rows, ant1.dtype)
        ant2_avg = np.empty(out_rows, ant2.dtype)

        centroid_avg = centroid_factory(out_rows, time_centroid)
        interval_avg = interval_factory(out_rows, interval)
        exposure_avg = exposure_factory(out_rows, exposure)

        for bli in range(nbl):
            off = bli*time_bins
            tbin = numba.uint32(0)
            nbin_values = numba.uint32(0)

            for ti in range(ntime):
                # Lookup input row for this baseline and time
                irow = in_lookup[bli, ti]

                if irow == -1:
                    continue

                # Lookup output row
                orow = out_lookup[off + tbin]

                nbin_values += 1

                # Here we can simply assign because we always have them
                time_avg[orow] = time_lookup[bli, tbin]
                ant1_avg[orow] = ant1[irow]
                ant2_avg[orow] = ant2[irow]

                centroid_adder(centroid_avg, orow, time_centroid, irow)
                interval_adder(interval_avg, orow, interval, irow)
                exposure_adder(exposure_avg, orow, exposure, irow)

                if nbin_values == time_bin_size:
                    centroid_normaliser(centroid_avg, orow, time_bin_size)
                    tbin += 1
                    nbin_values = numba.uint32(0)

            if nbin_values > 0:
                centroid_normaliser(centroid_avg, orow, nbin_values)

                tbin += 1

        return (time_avg, ant1_avg, ant2_avg,
                centroid_avg, interval_avg, exposure_avg)

    return impl
