# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numba

from africanus.averaging.row_mapping import row_mapper
from africanus.util.numba import is_numba_type_none


def output_factory(present):
    """ Returns function creating an output if present """

    if present:
        def impl(rows, array):
            return np.zeros((rows,) + array.shape[1:], array.dtype)
    else:
        def impl(rows, array):
            return None

    return numba.njit(nogil=True, cache=True)(impl)


def add_factory(present):
    """ Returns function for adding data to a bin """
    if present:
        def impl(output, orow, input, irow):
            output[orow] += input[irow]
    else:
        def impl(input, irow, output, orow):
            pass

    return numba.njit(nogil=True, cache=True)(impl)


def normaliser_factory(present):
    """ Returns function for normalising data in a bin """
    if present:
        def impl(data, idx, bin_size):
            data[idx] /= bin_size
    else:
        def impl(data, idx, bin_size):
            pass

    return numba.njit(nogil=True, cache=True)(impl)


@numba.generated_jit(nopython=True, nogil=True, cache=True)
def row_average(metadata, ant1, ant2,
                uvw=None, time=None, interval=None,
                weight=None, sigma=None):

    have_uvw = not is_numba_type_none(uvw)
    have_time = not is_numba_type_none(time)
    have_interval = not is_numba_type_none(interval)
    have_weight = not is_numba_type_none(weight)
    have_sigma = not is_numba_type_none(sigma)

    uvw_factory = output_factory(have_uvw)
    time_factory = output_factory(have_time)
    interval_factory = output_factory(have_interval)
    weight_factory = output_factory(have_weight)
    sigma_factory = output_factory(have_sigma)

    uvw_adder = add_factory(have_uvw)
    centroid_adder = add_factory(have_time)
    interval_adder = add_factory(have_interval)
    weight_adder = add_factory(have_weight)
    sigma_adder = add_factory(have_sigma)

    uvw_normaliser = normaliser_factory(have_uvw)
    centroid_normaliser = normaliser_factory(have_time)
    weight_normaliser = normaliser_factory(have_weight)
    sigma_normaliser = normaliser_factory(have_sigma)

    def impl(metadata, ant1, ant2,
             uvw=None, time=None, interval=None,
             weight=None, sigma=None):

        row_lookup, centroid_avg, exposure_sum = metadata
        out_rows = centroid_avg.shape[0]

        counts = np.zeros(out_rows, dtype=np.uint32)

        # These outputs are always present
        ant1_avg = np.empty(out_rows, ant1.dtype)
        ant2_avg = np.empty(out_rows, ant2.dtype)

        # Possibly present outputs for possibly present inputs
        uvw_avg = uvw_factory(out_rows, uvw)
        time_avg = time_factory(out_rows, time)
        interval_avg = interval_factory(out_rows, interval)
        weight_avg = weight_factory(out_rows, weight)
        sigma_avg = sigma_factory(out_rows, sigma)

        # Iterate over input rows, accumulating into output rows
        for i in range(row_lookup.shape[1]):
            in_row = row_lookup[0, i]
            out_row = row_lookup[1, i]
            counts[out_row] += 1

            # Here we can simply assign because input_row baselines
            # should always match output row baselines
            ant1_avg[out_row] = ant1[in_row]
            ant2_avg[out_row] = ant2[in_row]

            # Defer to functions for possibly missing input
            uvw_adder(uvw_avg, out_row, uvw, in_row)
            centroid_adder(time_avg, out_row, time, in_row)
            interval_adder(interval_avg, out_row, interval, in_row)
            weight_adder(weight_avg, out_row, weight, in_row)
            sigma_adder(sigma_avg, out_row, sigma, in_row)

        # Normalise
        for out_row in range(out_rows):
            count = counts[out_row]

            uvw_normaliser(uvw_avg, out_row, count)
            centroid_normaliser(time_avg, out_row, count)
            weight_normaliser(weight_avg, out_row, count)
            sigma_normaliser(sigma_avg, out_row, count)

        return (centroid_avg, exposure_sum,
                ant1_avg, ant2_avg,
                uvw_avg, time_avg, interval_avg,
                weight_avg, sigma_avg)

    return impl


@numba.generated_jit(nopython=True, nogil=True, cache=True)
def time_and_channel_average(time_centroid, exposure, ant1, ant2,
                             flag_row=None, uvw=None,
                             time=None, interval=None,
                             weight=None, sigma=None,
                             time_bin_secs=1.0):

    def impl(time_centroid, exposure, ant1, ant2,
             flag_row=None, uvw=None,
             time=None, interval=None,
             weight=None, sigma=None,
             time_bin_secs=1.0):

        metadata = row_mapper(time_centroid, exposure,
                              ant1, ant2, flag_row,
                              time_bin_secs)

        res = row_average(metadata, ant1, ant2, uvw)

    return impl

