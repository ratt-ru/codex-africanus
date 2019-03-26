# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

from numba import types
import numpy as np

from africanus.averaging.time_and_channel_mapping import (row_mapper,
                                                          channel_mapper)
from africanus.util.docs import DocstringTemplate
from africanus.util.numba import is_numba_type_none, generated_jit, njit


def output_factory(present):
    """ Returns function creating an output if present """

    if present:
        def impl(rows, array):
            return np.zeros((rows,) + array.shape[1:], array.dtype)
    else:
        def impl(rows, array):
            return None

    return njit(nogil=True, cache=True)(impl)


def add_factory(present):
    """ Returns function for adding data to a bin """
    if present:
        def impl(output, orow, input, irow):
            output[orow] += input[irow]
    else:
        def impl(input, irow, output, orow):
            pass

    return njit(nogil=True, cache=True)(impl)


def normaliser_factory(present):
    """ Returns function for normalising data in a bin """
    if present:
        def impl(data, idx, bin_size):
            data[idx] /= bin_size
    else:
        def impl(data, idx, bin_size):
            pass

    return njit(nogil=True, cache=True)(impl)


def matching_flag_factory(present):
    if present:
        def impl(flag_row, ri, out_flag_row, ro):
            return flag_row[ri] == out_flag_row[ro]
    else:
        def impl(flag_row, ri, out_flag_row, ro):
            return True

    return njit(nogil=True, cache=True)(impl)


_row_output_fields = ["antenna1", "antenna2", "time", "interval",
                      "uvw", "weight", "sigma"]
RowAverageOutput = namedtuple("RowAverageOutput", _row_output_fields)


@generated_jit(nopython=True, nogil=True, cache=True)
def row_average(meta, ant1, ant2, flag_row=None,
                time=None, interval=None, uvw=None,
                weight=None, sigma=None):

    have_flag_row = not is_numba_type_none(flag_row)
    have_uvw = not is_numba_type_none(uvw)
    have_time = not is_numba_type_none(time)
    have_interval = not is_numba_type_none(interval)
    have_weight = not is_numba_type_none(weight)
    have_sigma = not is_numba_type_none(sigma)

    flags_match = matching_flag_factory(have_flag_row)

    uvw_factory = output_factory(have_uvw)
    time_factory = output_factory(have_time)
    interval_factory = output_factory(have_interval)
    weight_factory = output_factory(have_weight)
    sigma_factory = output_factory(have_sigma)

    uvw_adder = add_factory(have_uvw)
    time_adder = add_factory(have_time)
    interval_adder = add_factory(have_interval)
    weight_adder = add_factory(have_weight)
    sigma_adder = add_factory(have_sigma)

    uvw_normaliser = normaliser_factory(have_uvw)
    time_normaliser = normaliser_factory(have_time)
    weight_normaliser = normaliser_factory(have_weight)
    sigma_normaliser = normaliser_factory(have_sigma)

    def impl(meta, ant1, ant2, flag_row=None,
             time=None, interval=None, uvw=None,
             weight=None, sigma=None):

        out_rows = meta.time_centroid.shape[0]

        counts = np.zeros(out_rows, dtype=np.uint32)
        combined_counts = np.zeros(out_rows, dtype=np.uint32)

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
        for in_row, out_row in enumerate(meta.map):
            # Input and output flags must match in order for the
            # current row to contribute to these columns
            if flags_match(flag_row, in_row, meta.flag_row, out_row):
                uvw_adder(uvw_avg, out_row, uvw, in_row)
                weight_adder(weight_avg, out_row, weight, in_row)
                sigma_adder(sigma_avg, out_row, sigma, in_row)
                counts[out_row] += 1

            # But these columns always included both
            # flagged and unflagged data
            time_adder(time_avg, out_row, time, in_row)
            interval_adder(interval_avg, out_row, interval, in_row)
            combined_counts[out_row] += 1

            # Here we can simply assign because input_row baselines
            # should always match output row baselines
            ant1_avg[out_row] = ant1[in_row]
            ant2_avg[out_row] = ant2[in_row]

        # Normalise
        for out_row in range(out_rows):
            count = counts[out_row]
            combined_count = combined_counts[out_row]

            if count > 0:
                uvw_normaliser(uvw_avg, out_row, count)
                weight_normaliser(weight_avg, out_row, count)
                sigma_normaliser(sigma_avg, out_row, count)

            if combined_count > 0:
                time_normaliser(time_avg, out_row, combined_count)

        return RowAverageOutput(ant1_avg, ant2_avg,
                                time_avg, interval_avg, uvw_avg,
                                weight_avg, sigma_avg)

    return impl


def vis_ampl_output_factory(present):
    """ Returns function producing output vis amplitudes if present """
    if present:
        def impl(shape, array):
            return np.zeros(shape, dtype=array.real.dtype)
    else:
        def impl(shape, array):
            pass

    return njit(nogil=True, cache=True)(impl)


def chan_output_factory(present):
    """ Returns function producing outputs if the array is present """
    if present:
        def impl(shape, array):
            return np.zeros(shape, dtype=array.dtype)
    else:
        def impl(shape, array):
            pass

    return njit(nogil=True, cache=True)(impl)


def vis_add_factory(present):
    """ Returns function adding visibilities and their amplitudes to a bin """
    if present:
        def impl(out_vis, out_vis_ampl, in_vis,
                 orow, ochan, irow, ichan, corr):
            iv = in_vis[irow, ichan, corr]
            out_vis[orow, ochan, corr] += iv
            out_vis_ampl[orow, ochan, corr] += np.abs(iv)
    else:
        def impl(out_vis, out_vis_ampl, in_vis,
                 orow, ochan, irow, ichan, corr):
            pass

    return njit(nogil=True, cache=True)(impl)


def chan_add_factory(present):
    """ Returns function for adding data to a bin """
    if present:
        def impl(output, input, orow, ochan, irow, ichan, corr):
            output[orow, ochan, corr] += input[irow, ichan, corr]
    else:
        def impl(output, input, orow, ochan, irow, ichan, corr):
            pass

    return njit(nogil=True, cache=True)(impl)


def vis_normalizer_factory(present):
    if present:
        def impl(vis_out, vis_in, row, chan, corr, vis_ampl):
            ampl = vis_ampl[row, chan, corr]

            if ampl != 0.0:
                vis_out[row, chan, corr] = vis_in[row, chan, corr] / ampl
    else:
        def impl(vis_out, vis_in, row, chan, corr, vis_ampl):
            pass

    return njit(nogil=True, cache=True)(impl)


def chan_normalizer_factory(present):
    """ Returns function normalising channel data in a bin """
    if present:
        def impl(data_out, data_in, row, chan, corr, bin_size):
            data_out[row, chan, corr] = data_in[row, chan, corr] / bin_size
    else:
        def impl(data_out, data_in, row, chan, corr, bin_size):
            pass

    return njit(nogil=True, cache=True)(impl)


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

    return njit(nogil=True, cache=True)(impl)


def is_chan_flagged_factory(present):
    if present:
        def impl(flag, r, f, c):
            return flag[r, f, c]
    else:
        def impl(flag, r, f, c):
            return False

    return njit(nogil=True, cache=True)(impl)


def set_flagged_factory(present):
    if present:
        def impl(flag, r, f, c):
            flag[r, f, c] = 1
    else:
        def impl(flag, r, f, c):
            pass

    return njit(nogil=True, cache=True)(impl)


_rowchan_output_fields = ["vis", "flag", "weight_spectrum", "sigma_spectrum"]
RowChanAverageOutput = namedtuple("RowChanAverageOutput",
                                  _rowchan_output_fields)


class RowChannelAverageException(Exception):
    pass


@generated_jit(nopython=True, nogil=True, cache=True)
def row_chan_average(row_meta, chan_meta, flag_row=None,
                     vis=None, flag=None,
                     weight_spectrum=None, sigma_spectrum=None,
                     chan_bin_size=1):

    have_flag_row = not is_numba_type_none(flag_row)
    have_vis = not is_numba_type_none(vis)
    have_flag = not is_numba_type_none(flag)
    have_weight = not is_numba_type_none(weight_spectrum)
    have_sigma = not is_numba_type_none(sigma_spectrum)

    flags_match = matching_flag_factory(have_flag_row)
    is_chan_flagged = is_chan_flagged_factory(have_flag)

    vis_factory = chan_output_factory(have_vis)
    vis_ampl_factory = vis_ampl_output_factory(have_vis)
    flag_factory = chan_output_factory(have_flag)
    weight_factory = chan_output_factory(have_weight)
    sigma_factory = chan_output_factory(have_sigma)

    vis_adder = vis_add_factory(have_vis)
    weight_adder = chan_add_factory(have_weight)
    sigma_adder = chan_add_factory(have_sigma)

    vis_normaliser = vis_normalizer_factory(have_vis)
    weight_normaliser = chan_normalizer_factory(have_weight)
    sigma_normaliser = chan_normalizer_factory(have_sigma)

    set_flagged = set_flagged_factory(have_flag)

    chan_corrs = chan_corr_factory(have_vis, have_flag,
                                   have_weight, have_sigma)

    def impl(row_meta, chan_meta, flag_row=None,
             vis=None, flag=None,
             weight_spectrum=None, sigma_spectrum=None,
             chan_bin_size=1):

        out_rows = row_meta.time_centroid.shape[0]
        chan_map, out_chans = chan_meta
        _, ncorrs = chan_corrs(vis, flag, weight_spectrum, sigma_spectrum)

        out_shape = (out_rows, out_chans, ncorrs)

        vis_avg = vis_factory(out_shape, vis)
        vis_ampl_avg = vis_ampl_factory(out_shape, vis)
        weight_spectrum_avg = weight_factory(out_shape, weight_spectrum)
        sigma_spectrum_avg = sigma_factory(out_shape, sigma_spectrum)

        flagged_vis_avg = vis_factory(out_shape, vis)
        flagged_vis_ampl_avg = vis_ampl_factory(out_shape, vis)
        flagged_weight_spectrum_avg = weight_factory(out_shape,
                                                     weight_spectrum)
        flagged_sigma_spectrum_avg = sigma_factory(out_shape,
                                                   sigma_spectrum)

        flag_avg = flag_factory(out_shape, flag)

        counts = np.zeros(out_shape, dtype=np.uint32)
        flag_counts = np.zeros(out_shape, dtype=np.uint32)

        # Iterate over input rows, accumulating into output rows
        for in_row, out_row in enumerate(row_meta.map):
            # TIME_CENTROID/EXPOSURE case applies here,
            # must have flagged input and output OR unflagged input and output
            if not flags_match(flag_row, in_row, row_meta.flag_row, out_row):
                continue

            for in_chan, out_chan in enumerate(chan_map):
                for corr in range(ncorrs):
                    if is_chan_flagged(flag, in_row, in_chan, corr):
                        # Increment flagged averages and counts
                        flag_counts[out_row, out_chan, corr] += 1

                        vis_adder(flagged_vis_avg,
                                  flagged_vis_ampl_avg,
                                  vis,
                                  out_row, out_chan, in_row, in_chan, corr)
                        weight_adder(flagged_weight_spectrum_avg,
                                     weight_spectrum,
                                     out_row, out_chan, in_row, in_chan, corr)
                        sigma_adder(flagged_sigma_spectrum_avg,
                                    sigma_spectrum,
                                    out_row, out_chan, in_row, in_chan, corr)
                    else:
                        # Increment unflagged averages and counts
                        counts[out_row, out_chan, corr] += 1

                        vis_adder(vis_avg, vis_ampl_avg, vis,
                                  out_row, out_chan, in_row, in_chan, corr)
                        weight_adder(weight_spectrum_avg, weight_spectrum,
                                     out_row, out_chan, in_row, in_chan, corr)
                        sigma_adder(sigma_spectrum_avg, sigma_spectrum,
                                    out_row, out_chan, in_row, in_chan, corr)

        for r in range(out_rows):
            for f in range(out_chans):
                for c in range(ncorrs):
                    count = counts[r, f, c]
                    flag_count = flag_counts[r, f, c]

                    if count > 0:
                        # We have some unflagged samples and
                        # only these are used as averaged output
                        vis_normaliser(vis_avg, vis_avg,
                                       r, f, c, vis_ampl_avg)
                        weight_normaliser(weight_spectrum_avg,
                                          weight_spectrum_avg,
                                          r, f, c, count)
                        sigma_normaliser(sigma_spectrum_avg,
                                         sigma_spectrum_avg,
                                         r, f, c, count)
                    elif flag_count > 0:
                        # We only have flagged samples and
                        # these are used as averaged output
                        vis_normaliser(vis_avg, flagged_vis_avg,
                                       r, f, c,
                                       flagged_vis_ampl_avg)
                        weight_normaliser(weight_spectrum_avg,
                                          flagged_weight_spectrum_avg,
                                          r, f, c, flag_count)

                        sigma_normaliser(sigma_spectrum_avg,
                                         flagged_sigma_spectrum_avg,
                                         r, f, c, flag_count)

                        # Flag the output bin
                        set_flagged(flag_avg, r, f, c)
                    else:
                        raise RowChannelAverageException("Zero-filled bin")

        return RowChanAverageOutput(vis_avg, flag_avg, weight_spectrum_avg,
                                    sigma_spectrum_avg)

    return impl


AverageOutput = namedtuple("AverageOutput",
                           ["time_centroid", "exposure", "flag_row"] +
                           _row_output_fields + _rowchan_output_fields)


@generated_jit(nopython=True, nogil=True, cache=True)
def merge_flags(flag_row, flag):
    have_flag_row = not is_numba_type_none(flag_row)
    have_flag = not is_numba_type_none(flag)

    if have_flag_row and have_flag:
        def impl(flag_row, flag):
            """ Check flag_row and flag agree """
            for r in range(flag.shape[0]):
                all_flagged = True

                for f in range(flag.shape[1]):
                    for c in range(flag.shape[2]):
                        if flag[r, f, c] == 0:
                            all_flagged = False
                            break

                    if not all_flagged:
                        break

                if (flag_row[r] != 0) != all_flagged:
                    raise ValueError("flag_row and flag arrays mismatch")

            return flag_row

    elif have_flag_row and not have_flag:
        def impl(flag_row, flag):
            """ Return flag_row """
            return flag_row

    elif not have_flag_row and have_flag:
        def impl(flag_row, flag):
            """ Construct flag_row from flag """
            new_flag_row = np.empty(flag.shape[0], dtype=flag.dtype)

            for r in range(flag.shape[0]):
                all_flagged = True

                for f in range(flag.shape[1]):
                    for c in range(flag.shape[2]):
                        if flag[r, f, c] == 0:
                            all_flagged = False
                            break

                    if not all_flagged:
                        break

                new_flag_row[r] = (1 if all_flagged else 0)

            return new_flag_row

    else:
        def impl(flag_row, flag):
            return None

    return impl


@generated_jit(nopython=True, nogil=True, cache=True)
def time_and_channel(time_centroid, exposure, antenna1, antenna2,
                     time=None, interval=None, flag_row=None,
                     uvw=None, weight=None, sigma=None,
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

    def impl(time_centroid, exposure, antenna1, antenna2,
             time=None, interval=None, flag_row=None,
             uvw=None, weight=None, sigma=None,
             vis=None, flag=None,
             weight_spectrum=None, sigma_spectrum=None,
             time_bin_secs=1.0, chan_bin_size=1):

        # Get the number of channels + correlations
        nchan, ncorr = chan_corrs(vis, flag, weight_spectrum, sigma_spectrum)

        # Merge flag_row and flag arrays
        flag_row = merge_flags(flag_row, flag)

        # Generate row mapping metadata
        row_meta = row_mapper(time_centroid, exposure, antenna1, antenna2,
                              flag_row=flag_row, time_bin_secs=time_bin_secs)

        # Generate channel mapping metadata
        chan_meta = channel_mapper(nchan, chan_bin_size)

        # Average row data
        row_data = row_average(row_meta, antenna1, antenna2, flag_row=flag_row,
                               time=time, interval=interval, uvw=uvw,
                               weight=weight, sigma=sigma)

        # Average channel data
        chan_data = row_chan_average(row_meta, chan_meta,
                                     flag_row=flag_row, vis=vis, flag=flag,
                                     weight_spectrum=weight_spectrum,
                                     sigma_spectrum=sigma_spectrum,
                                     chan_bin_size=chan_bin_size)

        # Have to explicitly write it out because numba tuples
        # are highly constrained types
        return AverageOutput(row_meta.time_centroid,
                             row_meta.exposure,
                             row_meta.flag_row,
                             row_data.antenna1,
                             row_data.antenna2,
                             row_data.time,
                             row_data.interval,
                             row_data.uvw,
                             row_data.weight,
                             row_data.sigma,
                             chan_data.vis,
                             chan_data.flag,
                             chan_data.weight_spectrum,
                             chan_data.sigma_spectrum)

    return impl


AVERAGING_DOCS = DocstringTemplate("""
Averages in time and channel.

Parameters
----------
time_centroid : $(array_type)
    Time centroid values of shape :code:`(row,)`
exposure : $(array_type)
    Exposure values of shape :code:`(row,)`
antenna1 : $(array_type)
    First antenna indices of shape :code:`(row,)`
antenna2 : $(array_type)
    Second antenna indices of shape :code:`(row,)`
time : $(array_type), optional
    Time values of shape :code:`(row,)`.
flag_row : $(array_type), optional
    Flagged rows of shape :code:`(row,)`.
uvw : $(array_type), optional
    UVW coordinates of shape :code:`(row, 3)`.
weight : $(array_type), optional
    Weight values of shape :code:`(row, corr)`.
sigma : $(array_type), optional
    Sigma values of shape :code:`(row, corr)`.
vis : $(array_type), optional
    Visibility data of shape :code:`(row, chan, corr)`.
flag : $(array_type), optional
    Flag data of shape :code:`(row, chan, corr)`.
weight_spectrum : $(array_type), optional
    Weight spectrum of shape :code:`(row, chan, corr)`.
sigma_spectrum : $(array_type), optional
    Sigma spectrum of shape :code:`(row, chan, corr)`.
time_bin_secs : float, optional
    Number of seconds of exposure to include in a bin.
    Defaults to 1.0.
chan_bin_size : int, optional
    Number of bins to average together.
    Defaults to 1.

Returns
-------
tuple
    Returns a namedtuple whose entries correspond to the input arrays.
""")


try:
    time_and_channel.__doc__ = AVERAGING_DOCS.subsititute(
                                    array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass
