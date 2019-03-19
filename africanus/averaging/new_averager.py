# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numba
from numba import types
import numpy as np

from africanus.averaging.row_mapping import row_mapper
from africanus.averaging.channel_mapping import channel_mapper
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


def chan_output_factory(present):
    if present:
        def impl(shape, array):
            return np.zeros(shape, dtype=array.dtype)
    else:
        def impl(shape, array):
            pass

    return numba.njit(nogil=True, cache=True)(impl)


def chan_add_factory(present):
    """ Returns function for adding data to a bin """
    if present:
        def impl(output, orow, ochan, input, irow, ichan, corr):
            output[orow, ochan, corr] += input[irow, ichan, corr]
    else:
        def impl(output, orow, ochan, input, irow, ichan, corr):
            pass

    return numba.njit(nogil=True, cache=True)(impl)


def chan_normalizer_factory(present):
    """ Returns function normalising channel data in a bin """
    if present:
        def impl(data, row, chan, corr, bin_size):
            data[row, chan, corr] /= bin_size
    else:
        def impl(data, row, chan, corr, bin_size):
            pass

    return numba.njit(nogil=True, cache=True)(impl)


def chan_corr_factory(have_vis, have_flag, have_weight, have_sigma):
    """ Returns function returning number of channels and correlations """
    if have_vis:
        def impl(vis, flag, weight_spectrum, sigma_spectrum):
            return vis.shape[1:]
    elif have_flag:
        def impl(vis, flag, weight_spectrum, sigma_spectrum):
            return flag.shape[1:]
    elif have_weight:
        def impl(vis, flag, weight_spectrum, sigma_spectrum):
            return weight_spectrum.shape[1:]
    elif have_sigma:
        def impl(vis, flag, weight_spectrum, sigma_spectrum):
            return sigma_spectrum.shape[1:]
    else:
        def impl(vis, flag, weight_spectrum, sigma_spectrum):
            return (1, 1)

    return numba.njit(nogil=True, cache=True)(impl)


@numba.generated_jit(nopython=True, nogil=True, cache=True)
def row_chan_average(row_meta, chan_meta, vis=None, flag=None,
                     weight_spectrum=None, sigma_spectrum=None,
                     chan_bin_size=1):

    have_vis = not is_numba_type_none(vis)
    have_flag = not is_numba_type_none(flag)
    have_weight = not is_numba_type_none(weight_spectrum)
    have_sigma = not is_numba_type_none(sigma_spectrum)

    vis_factory = chan_output_factory(have_vis)
    flag_factory = chan_output_factory(have_flag)
    weight_factory = chan_output_factory(have_weight)
    sigma_factory = chan_output_factory(have_sigma)

    vis_adder = chan_add_factory(have_vis)
    flag_adder = chan_add_factory(have_flag)
    weight_adder = chan_add_factory(have_weight)
    sigma_adder = chan_add_factory(have_sigma)

    vis_normaliser = chan_normalizer_factory(have_vis)
    flag_normaliser = chan_normalizer_factory(have_flag)
    weight_normaliser = chan_normalizer_factory(have_weight)
    sigma_normaliser = chan_normalizer_factory(have_sigma)

    chan_corrs = chan_corr_factory(have_vis, have_flag,
                                   have_weight, have_sigma)

    def impl(row_meta, chan_meta, vis=None, flag=None,
             weight_spectrum=None, sigma_spectrum=None,
             chan_bin_size=1):

        row_lookup, centroid_avg, _ = row_meta
        out_rows = centroid_avg.shape[0]
        chan_map, out_chans = chan_meta
        _, ncorrs = chan_corrs(vis, flag, weight_spectrum, sigma_spectrum)

        out_shape = (out_rows, out_chans, ncorrs)
        vis_avg = vis_factory(out_shape, vis)
        flag_avg = flag_factory(out_shape, flag)
        weight_spectrum_avg = weight_factory(out_shape, weight_spectrum)
        sigma_spectrum_avg = sigma_factory(out_shape, sigma_spectrum)

        counts = np.zeros(out_shape, dtype=np.uint32)

        # Iterate over input rows, accumulating into output rows
        for i in range(row_lookup.shape[1]):
            in_row = row_lookup[0, i]
            out_row = row_lookup[1, i]

            for in_chan, out_chan in enumerate(chan_map):
                for c in range(ncorrs):
                    counts[out_row, out_chan, c] += 1

                    vis_adder(vis_avg, out_row, out_chan,
                              vis, in_row, in_chan, c)
                    flag_adder(flag_avg, out_row, out_chan,
                               flag, in_row, in_chan, c)
                    weight_adder(weight_spectrum_avg, out_row, out_chan,
                                 weight_spectrum, in_row, in_chan, c)
                    sigma_adder(sigma_spectrum_avg, out_row, out_chan,
                                sigma_spectrum, in_row, in_chan, c)

        for r in range(out_rows):
            for f in range(out_chans):
                for c in range(ncorrs):
                    count = counts[r, f, c]

                    vis_normaliser(vis_avg, r, f, c, count)
                    flag_normaliser(flag_avg, r, f, c, count)
                    weight_normaliser(weight_spectrum_avg, r, f, c, count)
                    sigma_normaliser(sigma_spectrum_avg, r, f, c, count)

        return vis_avg, flag_avg, weight_spectrum_avg, sigma_spectrum_avg

    return impl


@numba.generated_jit(nopython=True, nogil=True, cache=True)
def time_and_channel_average(time_centroid, exposure, ant1, ant2,
                             flag_row=None, uvw=None, time=None, interval=None,
                             weight=None, sigma=None,
                             vis=None, flag=None,
                             weight_spectrum=None, sigma_spectrum=None,
                             time_bin_secs=1.0, chan_bin_size=1):

    valid_types = (types.misc.Omitted, types.scalars.Float,
                   types.scalars.Integer)

    if not isinstance(time_bin_secs, valid_types):
        raise TypeError("time_bin_secs must be a scalar float")

    valid_types = (types.misc.Omitted, types.scalars.Integer)

    if not isinstance(chan_bin_size, valid_types):
        raise TypeError("chan_bin_size must be a scalar integer")

    have_vis = not is_numba_type_none(vis)
    have_flag = not is_numba_type_none(flag)
    have_weight = not is_numba_type_none(weight_spectrum)
    have_sigma = not is_numba_type_none(sigma_spectrum)

    chan_corrs = chan_corr_factory(have_vis, have_flag,
                                   have_weight, have_sigma)

    def impl(time_centroid, exposure, ant1, ant2,
             flag_row=None, uvw=None, time=None, interval=None,
             weight=None, sigma=None,
             vis=None, flag=None,
             weight_spectrum=None, sigma_spectrum=None,
             time_bin_secs=1.0, chan_bin_size=1):

        row_meta = row_mapper(time_centroid, exposure,
                              ant1, ant2, flag_row,
                              time_bin_secs)

        nchan, ncorr = chan_corrs(vis, flag, weight_spectrum, sigma_spectrum)
        chan_meta = channel_mapper(nchan, chan_bin_size)

        row_data = row_average(row_meta, ant1, ant2, uvw,
                               time, interval, weight, sigma)

        chan_data = row_chan_average(row_meta, chan_meta,
                                     vis=vis, flag=flag,
                                     weight_spectrum=weight_spectrum,
                                     sigma_spectrum=sigma_spectrum,
                                     chan_bin_size=chan_bin_size)

        return row_data, chan_data

    return impl
