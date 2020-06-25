# -*- coding: utf-8 -*-


from collections import namedtuple

from numba import types
import numpy as np

from africanus.averaging.time_and_channel_mapping import (row_mapper,
                                                          channel_mapper)
from africanus.averaging.shared import (chan_corrs,
                                        flags_match,
                                        merge_flags)

from africanus.util.docs import DocstringTemplate
from africanus.util.numba import (is_numba_type_none, generated_jit,
                                  njit, overload)

TUPLE_TYPE = 0
ARRAY_TYPE = 1
NONE_TYPE = 2


def matching_flag_factory(present):
    if present:
        def impl(flag_row, ri, out_flag_row, ro):
            return flag_row[ri] == out_flag_row[ro]
    else:
        def impl(flag_row, ri, out_flag_row, ro):
            return True

    return njit(nogil=True, cache=True, inline='always')(impl)


def is_chan_flagged(flag, r, f, c):
    pass


@overload(is_chan_flagged, inline='always')
def _is_chan_flagged(flag, r, f, c):
    if is_numba_type_none(flag):
        def impl(flag, r, f, c):
            return True
    else:
        def impl(flag, r, f, c):
            return flag[r, f, c]

    return impl


@njit(nogil=True, inline='always')
def chan_add(output, input, orow, ochan, irow, ichan, corr):
    if input is not None:
        output[orow, ochan, corr] += input[irow, ichan, corr]


_row_output_fields = ["antenna1", "antenna2", "time_centroid", "exposure",
                      "uvw", "weight", "sigma"]
RowAverageOutput = namedtuple("RowAverageOutput", _row_output_fields)


@generated_jit(nopython=True, nogil=True, cache=True)
def row_average(meta, ant1, ant2, flag_row=None,
                time_centroid=None, exposure=None, uvw=None,
                weight=None, sigma=None):

    have_flag_row = not is_numba_type_none(flag_row)
    flags_match = matching_flag_factory(have_flag_row)

    def impl(meta, ant1, ant2, flag_row=None,
             time_centroid=None, exposure=None, uvw=None,
             weight=None, sigma=None):

        out_rows = meta.time.shape[0]

        counts = np.zeros(out_rows, dtype=np.uint32)

        # These outputs are always present
        ant1_avg = np.empty(out_rows, ant1.dtype)
        ant2_avg = np.empty(out_rows, ant2.dtype)

        # Possibly present outputs for possibly present inputs
        uvw_avg = (
            None if uvw is None else
            np.zeros((out_rows,) + uvw.shape[1:],
                     dtype=uvw.dtype))

        time_centroid_avg = (
            None if time_centroid is None else
            np.zeros((out_rows,) + time_centroid.shape[1:],
                     dtype=time_centroid.dtype))

        exposure_avg = (
            None if exposure is None else
            np.zeros((out_rows,) + exposure.shape[1:],
                     dtype=exposure.dtype))

        weight_avg = (
            None if weight is None else
            np.zeros((out_rows,) + weight.shape[1:],
                     dtype=weight.dtype))

        sigma_avg = (
            None if sigma is None else
            np.zeros((out_rows,) + sigma.shape[1:],
                     dtype=sigma.dtype))

        sigma_weight_sum = (
            None if sigma is None else
            np.zeros((out_rows,) + sigma.shape[1:],
                     dtype=sigma.dtype))

        # Iterate over input rows, accumulating into output rows
        for in_row, out_row in enumerate(meta.map):
            # Input and output flags must match in order for the
            # current row to contribute to these columns
            if flags_match(flag_row, in_row, meta.flag_row, out_row):
                if uvw is not None:
                    uvw_avg[out_row, 0] += uvw[in_row, 0]
                    uvw_avg[out_row, 1] += uvw[in_row, 1]
                    uvw_avg[out_row, 2] += uvw[in_row, 2]

                if time_centroid is not None:
                    time_centroid_avg[out_row] += time_centroid[in_row]

                if exposure is not None:
                    exposure_avg[out_row] += exposure[in_row]

                if weight is not None:
                    for co in range(weight.shape[1]):
                        weight_avg[out_row, co] += weight[in_row, co]

                if sigma is not None:
                    for co in range(sigma.shape[1]):
                        sva = sigma[in_row, co]**2

                        # Use provided weights
                        if weight is not None:
                            wt = weight[in_row, co]
                            sva *= wt ** 2
                            sigma_weight_sum[out_row, co] += wt
                        # Natural weights
                        else:
                            sigma_weight_sum[out_row, co] += 1.0

                        # Assign
                        sigma_avg[out_row, co] += sva

                counts[out_row] += 1

            # Here we can simply assign because input_row baselines
            # should always match output row baselines
            ant1_avg[out_row] = ant1[in_row]
            ant2_avg[out_row] = ant2[in_row]

        # Normalise
        for out_row in range(out_rows):
            count = counts[out_row]

            if count > 0:
                # Normalise uvw
                if uvw is not None:
                    uvw_avg[out_row, 0] /= count
                    uvw_avg[out_row, 1] /= count
                    uvw_avg[out_row, 2] /= count

                # Normalise time centroid
                if time_centroid is not None:
                    time_centroid_avg[out_row] /= count

                # Normalise sigma
                if sigma is not None:
                    for co in range(sigma.shape[1]):
                        ssva = sigma_avg[out_row, co]
                        wt = sigma_weight_sum[out_row, co]

                        if wt != 0.0:
                            ssva /= (wt**2)

                        sigma_avg[out_row, co] = np.sqrt(ssva)

        return RowAverageOutput(ant1_avg, ant2_avg,
                                time_centroid_avg,
                                exposure_avg, uvw_avg,
                                weight_avg, sigma_avg)

    return impl


_rowchan_output_fields = ["vis", "flag", "weight_spectrum", "sigma_spectrum"]
RowChanAverageOutput = namedtuple("RowChanAverageOutput",
                                  _rowchan_output_fields)


class RowChannelAverageException(Exception):
    pass


@generated_jit(nopython=True, nogil=True, cache=True)
def row_chan_average(row_meta, chan_meta,
                     flag_row=None, weight=None,
                     vis=None, flag=None,
                     weight_spectrum=None, sigma_spectrum=None):

    dummy_chan_freq = None
    dummy_chan_width = None

    have_weight = not is_numba_type_none(weight)

    have_vis = not is_numba_type_none(vis)
    have_flag = not is_numba_type_none(flag)
    have_weight_spectrum = not is_numba_type_none(weight_spectrum)
    have_sigma_spectrum = not is_numba_type_none(sigma_spectrum)

    def impl(row_meta, chan_meta, flag_row=None, weight=None,
             vis=None, flag=None,
             weight_spectrum=None, sigma_spectrum=None):

        out_rows = row_meta.time.shape[0]
        nchan, ncorrs = chan_corrs(vis, flag,
                                   weight_spectrum, sigma_spectrum,
                                   dummy_chan_freq, dummy_chan_width,
                                   dummy_chan_width, dummy_chan_width)

        chan_map, out_chans = chan_meta

        out_shape = (out_rows, out_chans, ncorrs)

        # Visibility output and intermediate arrays
        if vis is None:
            vis_avg = None
            vis_weight_sum = None
            flagged_vis_avg = None
            flagged_vis_weight_sum = None
        else:
            vis_avg = np.zeros(out_shape, dtype=vis.dtype)
            vis_weight_sum = np.zeros(out_shape, dtype=vis.dtype)
            flagged_vis_avg = np.zeros_like(vis_avg)
            flagged_vis_weight_sum = np.zeros_like(vis_weight_sum)

        # Flagged output and intermediate arrays
        if flag is None:
            flag_avg = None
        else:
            flag_avg = np.zeros(out_shape, dtype=flag.dtype)

        # Weight spectrum output and intermediate arrays
        if weight_spectrum is None:
            weight_spectrum_avg = None
            flagged_weight_spectrum_avg = None
        else:
            weight_spectrum_avg = np.zeros(
                out_shape, dtype=weight_spectrum.dtype)
            flagged_weight_spectrum_avg = np.zeros_like(weight_spectrum_avg)

        # Sigma spectrum output and intermediate arrays
        if sigma_spectrum is None:
            sigma_spectrum_avg = None
            sigma_spectrum_weight_sum = None
            flagged_sigma_spectrum_avg = None
            flagged_sigma_spectrum_weight_sum = None
        else:
            sigma_spectrum_avg = np.zeros(
                out_shape, dtype=sigma_spectrum.dtype)
            sigma_spectrum_weight_sum = np.zeros_like(sigma_spectrum_avg)
            flagged_sigma_spectrum_avg = np.zeros_like(sigma_spectrum_avg)
            flagged_sigma_spectrum_weight_sum = np.zeros_like(
                sigma_spectrum_avg)

        counts = np.zeros(out_shape, dtype=np.uint32)
        flag_counts = np.zeros(out_shape, dtype=np.uint32)

        # Iterate over input rows, accumulating into output rows
        for ri, ro in enumerate(row_meta.map):
            # TIME_CENTROID/EXPOSURE case applies here,
            # must have flagged input and output OR unflagged input and output
            if not flags_match(flag_row, ri, row_meta.flag_row, ro):
                continue

            for fi, fo in enumerate(chan_map):
                for co in range(ncorrs):
                    flagged = have_flag and flag[ri, fi, co] != 0

                    if flagged:
                        flag_counts[ro, fo, co] += 1
                    else:
                        counts[ro, fo, co] += 1

                    # Aggregate visibilities
                    if have_vis:
                        # Use full-resolution weight spectrum if given
                        # else weights, else natural weights
                        wt = (weight_spectrum[ri, fi, co]
                              if have_weight_spectrum else
                              weight[ri, co] if have_weight else 1.0)

                        iv = vis[ri, fi, co] * wt

                        if flagged:
                            flagged_vis_avg[ro, fo, co] += iv
                            flagged_vis_weight_sum[ro, fo, co] += wt
                        else:
                            vis_avg[ro, fo, co] += iv
                            vis_weight_sum[ro, fo, co] += wt

                    # Weight Spectrum
                    if have_weight_spectrum:
                        if flagged:
                            flagged_weight_spectrum_avg[ro, fo, co] += (
                                weight_spectrum[ri, fi, co])
                        else:
                            weight_spectrum_avg[ro, fo, co] += (
                                weight_spectrum[ri, fi, co])

                    # Sigma Spectrum
                    if have_sigma_spectrum:
                        # Use full-resolution weight spectrum if given
                        # else weights, else natural weights
                        wt = (weight_spectrum[ri, fi, co]
                              if have_weight_spectrum else
                              weight[ri, co] if have_weight else 1.0)

                        ssin = sigma_spectrum[ri, fi, co]**2 * wt**2

                        if flagged:
                            flagged_sigma_spectrum_avg[ro, fo, co] += ssin
                            flagged_sigma_spectrum_weight_sum[ro, fo, co] += wt
                        else:
                            sigma_spectrum_avg[ro, fo, co] += ssin
                            sigma_spectrum_weight_sum[ro, fo, co] += wt

        for r in range(out_rows):
            for f in range(out_chans):
                for c in range(ncorrs):
                    if counts[r, f, c] > 0:
                        if have_vis:
                            vwsum = vis_weight_sum[r, f, c]
                            vin = vis_avg[r, f, c]

                        if have_sigma_spectrum:
                            sswsum = sigma_spectrum_weight_sum[r, f, c]
                            ssin = sigma_spectrum_avg[r, f, c]

                        flagged = 0
                    elif flag_counts[r, f, c] > 0:
                        if have_vis:
                            vwsum = flagged_vis_weight_sum[r, f, c]
                            vin = flagged_vis_avg[r, f, c]

                        if have_sigma_spectrum:
                            sswsum = flagged_sigma_spectrum_weight_sum[r, f, c]
                            ssin = flagged_sigma_spectrum_avg[r, f, c]

                        flagged = 1
                    else:
                        raise RowChannelAverageException("Zero-filled bin")

                    # Normalise visibilities
                    if have_vis and vwsum != 0.0:
                        vis_avg[r, f, c] = vin / vwsum

                    # Normalise Sigma Spectrum
                    if have_sigma_spectrum and sswsum != 0.0:
                        # sqrt(sigma**2 * weight**2 / (weight(sum**2)))
                        sigma_spectrum_avg[r, f, c] = np.sqrt(ssin / sswsum**2)

                    # Set flag
                    if have_flag:
                        flag_avg[r, f, c] = flagged

                    # Copy Weights if flagged
                    if have_weight_spectrum and flagged:
                        weight_spectrum_avg[r, f, c] = (
                            flagged_weight_spectrum_avg[r, f, c])

        return RowChanAverageOutput(vis_avg, flag_avg,
                                    weight_spectrum_avg,
                                    sigma_spectrum_avg)

    return impl


_chan_output_fields = ["chan_freq", "chan_width", "effective_bw", "resolution"]
ChannelAverageOutput = namedtuple("ChannelAverageOutput", _chan_output_fields)


@generated_jit(nopython=True, nogil=True, cache=True)
def chan_average(chan_meta, chan_freq=None, chan_width=None,
                 effective_bw=None, resolution=None):

    def impl(chan_meta, chan_freq=None, chan_width=None,
             effective_bw=None, resolution=None):
        chan_map, out_chans = chan_meta

        chan_freq_avg = (
            None if chan_freq is None else
            np.zeros(out_chans, dtype=chan_freq.dtype))

        chan_width_avg = (
            None if chan_width is None else
            np.zeros(out_chans, dtype=chan_width.dtype))

        effective_bw_avg = (
            None if effective_bw is None else
            np.zeros(out_chans, dtype=effective_bw.dtype))

        resolution_avg = (
            None if resolution is None else
            np.zeros(out_chans, dtype=resolution.dtype))

        counts = np.zeros(out_chans, dtype=np.uint32)

        for in_chan, out_chan in enumerate(chan_map):
            counts[out_chan] += 1

            if chan_freq is not None:
                chan_freq_avg[out_chan] += chan_freq[in_chan]

            if chan_width is not None:
                chan_width_avg[out_chan] += chan_width[in_chan]

            if effective_bw is not None:
                effective_bw_avg[out_chan] += effective_bw[in_chan]

            if resolution is not None:
                resolution_avg[out_chan] += resolution[in_chan]

        for out_chan in range(out_chans):
            if chan_freq is not None:
                chan_freq_avg[out_chan] /= counts[out_chan]

        return ChannelAverageOutput(chan_freq_avg, chan_width_avg,
                                    effective_bw_avg, resolution_avg)

    return impl


AverageOutput = namedtuple("AverageOutput",
                           ["time", "interval", "flag_row"] +
                           _row_output_fields +
                           _chan_output_fields +
                           _rowchan_output_fields)


@generated_jit(nopython=True, nogil=True, cache=True)
def time_and_channel(time, interval, antenna1, antenna2,
                     time_centroid=None, exposure=None, flag_row=None,
                     uvw=None, weight=None, sigma=None,
                     chan_freq=None, chan_width=None,
                     effective_bw=None, resolution=None,
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

    def impl(time, interval, antenna1, antenna2,
             time_centroid=None, exposure=None, flag_row=None,
             uvw=None, weight=None, sigma=None,
             chan_freq=None, chan_width=None,
             effective_bw=None, resolution=None,
             vis=None, flag=None,
             weight_spectrum=None, sigma_spectrum=None,
             time_bin_secs=1.0, chan_bin_size=1):

        nchan, ncorrs = chan_corrs(vis, flag,
                                   weight_spectrum, sigma_spectrum,
                                   chan_freq, chan_width,
                                   effective_bw, resolution)

        # Merge flag_row and flag arrays
        flag_row = merge_flags(flag_row, flag)

        # Generate row mapping metadata
        row_meta = row_mapper(time, interval, antenna1, antenna2,
                              flag_row=flag_row, time_bin_secs=time_bin_secs)

        # Generate channel mapping metadata
        chan_meta = channel_mapper(nchan, chan_bin_size)

        # Average row data
        row_data = row_average(row_meta, antenna1, antenna2, flag_row=flag_row,
                               time_centroid=time_centroid, exposure=exposure,
                               uvw=uvw, weight=weight, sigma=sigma)

        # Average channel data
        chan_data = chan_average(chan_meta, chan_freq=chan_freq,
                                 chan_width=chan_width,
                                 effective_bw=effective_bw,
                                 resolution=resolution)

        # Average row and channel data
        row_chan_data = row_chan_average(row_meta, chan_meta,
                                         flag_row=flag_row, weight=weight,
                                         vis=vis, flag=flag,
                                         weight_spectrum=weight_spectrum,
                                         sigma_spectrum=sigma_spectrum)

        # Have to explicitly write it out because numba tuples
        # are highly constrained types
        return AverageOutput(row_meta.time,
                             row_meta.interval,
                             row_meta.flag_row,
                             row_data.antenna1,
                             row_data.antenna2,
                             row_data.time_centroid,
                             row_data.exposure,
                             row_data.uvw,
                             row_data.weight,
                             row_data.sigma,
                             chan_data.chan_freq,
                             chan_data.chan_width,
                             chan_data.effective_bw,
                             chan_data.resolution,
                             row_chan_data.vis,
                             row_chan_data.flag,
                             row_chan_data.weight_spectrum,
                             row_chan_data.sigma_spectrum)

    return impl


AVERAGING_DOCS = DocstringTemplate("""
Averages in time and channel.

Parameters
----------
time : $(array_type)
    Time values of shape :code:`(row,)`.
interval : $(array_type)
    Interval values of shape :code:`(row,)`.
antenna1 : $(array_type)
    First antenna indices of shape :code:`(row,)`
antenna2 : $(array_type)
    Second antenna indices of shape :code:`(row,)`
time_centroid : $(array_type), optional
    Time centroid values of shape :code:`(row,)`
exposure : $(array_type), optional
    Exposure values of shape :code:`(row,)`
flag_row : $(array_type), optional
    Flagged rows of shape :code:`(row,)`.
uvw : $(array_type), optional
    UVW coordinates of shape :code:`(row, 3)`.
weight : $(array_type), optional
    Weight values of shape :code:`(row, corr)`.
sigma : $(array_type), optional
    Sigma values of shape :code:`(row, corr)`.
chan_freq : $(array_type), optional
    Channel frequencies of shape :code:`(chan,)`.
chan_width : $(array_type), optional
    Channel widths of shape :code:`(chan,)`.
effective_bw : $(array_type), optional
    Effective channel bandwidth of shape :code:`(chan,)`.
resolution : $(array_type), optional
    Effective channel resolution of shape :code:`(chan,)`.
vis : $(array_type), optional
    Visibility data of shape :code:`(row, chan, corr)`.
flag : $(array_type), optional
    Flag data of shape :code:`(row, chan, corr)`.
weight_spectrum : $(array_type), optional
    Weight spectrum of shape :code:`(row, chan, corr)`.
sigma_spectrum : $(array_type), optional
    Sigma spectrum of shape :code:`(row, chan, corr)`.
time_bin_secs : float, optional
    Maximum summed interval in seconds to include within a bin.
    Defaults to 1.0.
chan_bin_size : int, optional
    Number of bins to average together.
    Defaults to 1.

Notes
-----

The implementation currently requires unique lexicographical
combinations of (TIME, ANTENNA1, ANTENNA2). This can usually
be achieved by suitably partitioning input data on indexing rows,
DATA_DESC_ID and SCAN_NUMBER in particular.

Returns
-------
namedtuple
    A namedtuple whose entries correspond to the input arrays.
    Output arrays will be ``None`` if the inputs were ``None``.
""")


try:
    time_and_channel.__doc__ = AVERAGING_DOCS.substitute(
                                    array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass
