# -*- coding: utf-8 -*-


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


def comp_add_factory(present):
    """
    Returns function for adding data with components to a bin.
    Rows are assumed to be in the first dimension and
    components are assumed to be in the second
    """
    if present:
        def impl(output, orow, input, irow):
            for c in range(output.shape[1]):
                output[orow, c] += input[irow, c]
    else:
        def impl(input, irow, output, orow):
            pass

    return njit(nogil=True, cache=True)(impl)


def sigma_add_factory(have_sigma, have_weight):
    """
    Returns function for adding sigma values to a bin.
    Uses provided weights, else natural weights
    """
    if not have_sigma:
        def impl(out_sigma, out_weight_sum, orow, in_sigma, in_weight, irow):
            pass
    elif have_weight:
        def impl(out_sigma, out_weight_sum, orow, in_sigma, in_weight, irow):
            for c in range(out_sigma.shape[1]):
                out_sigma[orow, c] += (in_sigma[irow, c]**2 *
                                       in_weight[irow, c]**2)
                out_weight_sum[orow, c] += in_weight[irow, c]
    else:
        def impl(out_sigma, out_weight_sum, orow, in_sigma, in_weight, irow):
            for c in range(out_sigma.shape[1]):
                out_sigma[orow, c] += in_sigma[irow, c]**2
                out_weight_sum[orow, c] += in_weight[irow, c]

    return njit(nogil=True, cache=True)(impl)


def normaliser_factory(present):
    """ Returns function for normalising data in a bin """
    if present:
        def impl(data, row, bin_size):
            data[row] /= bin_size
    else:
        def impl(data, row, bin_size):
            pass

    return njit(nogil=True, cache=True)(impl)


def sigma_normaliser_factory(present):
    """ Returns function for normalising sigma in a bin """
    if present:
        def impl(sigma, row, weight_sum):
            for c in range(sigma.shape[1]):
                wt = weight_sum[row, c]

                if wt == 0.0:
                    continue

                sigma[row, c] = np.sqrt(sigma[row, c] / (wt**2))
    else:
        def impl(sigma, row, weight_sum):
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


_row_output_fields = ["antenna1", "antenna2", "time_centroid", "exposure",
                      "uvw", "weight", "sigma"]
RowAverageOutput = namedtuple("RowAverageOutput", _row_output_fields)


@generated_jit(nopython=True, nogil=True, cache=True)
def row_average(meta, ant1, ant2, flag_row=None,
                time_centroid=None, exposure=None, uvw=None,
                weight=None, sigma=None):

    have_flag_row = not is_numba_type_none(flag_row)
    have_uvw = not is_numba_type_none(uvw)
    have_time_centroid = not is_numba_type_none(time_centroid)
    have_exposure = not is_numba_type_none(exposure)
    have_weight = not is_numba_type_none(weight)
    have_sigma = not is_numba_type_none(sigma)

    flags_match = matching_flag_factory(have_flag_row)

    uvw_factory = output_factory(have_uvw)
    time_centroid_factory = output_factory(have_time_centroid)
    exposure_factory = output_factory(have_exposure)
    weight_factory = output_factory(have_weight)
    sigma_factory = output_factory(have_sigma)

    time_centroid_adder = add_factory(have_time_centroid)
    exposure_adder = add_factory(have_exposure)
    uvw_adder = comp_add_factory(have_uvw)
    weight_adder = comp_add_factory(have_weight)
    sigma_adder = sigma_add_factory(have_sigma, have_weight)

    uvw_normaliser = normaliser_factory(have_uvw)
    sigma_normaliser = sigma_normaliser_factory(have_sigma)
    time_centroid_normaliser = normaliser_factory(have_time_centroid)

    def impl(meta, ant1, ant2, flag_row=None,
             time_centroid=None, exposure=None, uvw=None,
             weight=None, sigma=None):

        out_rows = meta.time.shape[0]

        counts = np.zeros(out_rows, dtype=np.uint32)

        # These outputs are always present
        ant1_avg = np.empty(out_rows, ant1.dtype)
        ant2_avg = np.empty(out_rows, ant2.dtype)

        # Possibly present outputs for possibly present inputs
        uvw_avg = uvw_factory(out_rows, uvw)
        time_centroid_avg = time_centroid_factory(out_rows, time_centroid)
        exposure_avg = exposure_factory(out_rows, exposure)
        weight_avg = weight_factory(out_rows, weight)
        sigma_avg = sigma_factory(out_rows, sigma)
        sigma_weight_sum = sigma_factory(out_rows, sigma)

        # Iterate over input rows, accumulating into output rows
        for in_row, out_row in enumerate(meta.map):
            # Input and output flags must match in order for the
            # current row to contribute to these columns
            if flags_match(flag_row, in_row, meta.flag_row, out_row):
                uvw_adder(uvw_avg, out_row, uvw, in_row)
                weight_adder(weight_avg, out_row, weight, in_row)
                sigma_adder(sigma_avg, sigma_weight_sum, out_row,
                            sigma, weight, in_row)
                time_centroid_adder(time_centroid_avg, out_row,
                                    time_centroid, in_row)
                exposure_adder(exposure_avg, out_row, exposure, in_row)

                counts[out_row] += 1

            # Here we can simply assign because input_row baselines
            # should always match output row baselines
            ant1_avg[out_row] = ant1[in_row]
            ant2_avg[out_row] = ant2[in_row]

        # Normalise
        for out_row in range(out_rows):
            count = counts[out_row]

            if count > 0:
                uvw_normaliser(uvw_avg, out_row, count)
                time_centroid_normaliser(time_centroid_avg, out_row, count)
                sigma_normaliser(sigma_avg, out_row, sigma_weight_sum)

        return RowAverageOutput(ant1_avg, ant2_avg,
                                time_centroid_avg, exposure_avg, uvw_avg,
                                weight_avg, sigma_avg)

    return impl


def weight_sum_output_factory(present):
    """ Returns function producing vis weight sum if vis present """
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


def vis_add_factory(have_vis, have_weight, have_weight_spectrum):
    """ Returns function adding weighted visibilities to a bin """
    if not have_vis:
        def impl(out_vis, out_weight_sum, in_vis, weight, weight_spectrum,
                 orow, ochan, irow, ichan, corr):
            pass
    elif have_weight_spectrum:
        # Always prefer more accurate weight spectrum if we have it
        def impl(out_vis, out_weight_sum, in_vis,
                 weight, weight_spectrum,
                 orow, ochan, irow, ichan, corr):

            wt = weight_spectrum[irow, ichan, corr]
            iv = in_vis[irow, ichan, corr] * wt
            out_vis[orow, ochan, corr] += iv
            out_weight_sum[orow, ochan, corr] += wt

    elif have_weight:
        # Otherwise fall back to row weights
        def impl(out_vis, out_weight_sum, in_vis,
                 weight, weight_spectrum,
                 orow, ochan, irow, ichan, corr):

            wt = weight[irow, corr]
            iv = in_vis[irow, ichan, corr] * wt
            out_vis[orow, ochan, corr] += iv
            out_weight_sum[orow, ochan, corr] += wt
    else:
        # Natural weights
        def impl(out_vis, out_weight_sum, in_vis,
                 weight, weight_spectrum,
                 orow, ochan, irow, ichan, corr):

            iv = in_vis[irow, ichan, corr]
            out_vis[orow, ochan, corr] += iv
            out_weight_sum[orow, ochan, corr] += 1.0

    return njit(nogil=True, cache=True)(impl)


def sigma_spectrum_add_factory(have_vis, have_weight, have_weight_spectrum):
    """ Returns function adding weighted sigma to a bin """
    if not have_vis:
        def impl(out_sigma, out_weight_sum, in_sigma,
                 weight, weight_spectrum,
                 orow, ochan, irow, ichan, corr):
            pass

    elif have_weight_spectrum:
        # Always prefer more accurate weight spectrum if we have it
        def impl(out_sigma, out_weight_sum, in_sigma,
                 weight, weight_spectrum,
                 orow, ochan, irow, ichan, corr):

            # sum(sigma**2 * weight**2)
            wt = weight_spectrum[irow, ichan, corr]
            is_ = in_sigma[irow, ichan, corr]**2 * wt**2
            out_sigma[orow, ochan, corr] += is_
            out_weight_sum[orow, ochan, corr] += wt

    elif have_weight:
        # Otherwise fall back to row weights
        def impl(out_sigma, out_weight_sum, in_sigma,
                 weight, weight_spectrum,
                 orow, ochan, irow, ichan, corr):

            # sum(sigma**2 * weight**2)
            wt = weight[irow, corr]
            is_ = in_sigma[irow, ichan, corr]**2 * wt**2
            out_sigma[orow, ochan, corr] += is_
            out_weight_sum[orow, ochan, corr] += wt
    else:
        # Natural weights
        def impl(out_sigma, out_weight_sum, in_sigma,
                 weight, weight_spectrum,
                 orow, ochan, irow, ichan, corr):

            # sum(sigma**2 * weight**2)
            out_sigma[orow, ochan, corr] += in_sigma[irow, ichan, corr]**2
            out_weight_sum[orow, ochan, corr] += 1.0

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


def vis_normaliser_factory(present):
    if present:
        def impl(vis_out, vis_in, row, chan, corr, weight_sum):
            wsum = weight_sum[row, chan, corr]

            if wsum != 0.0:
                vis_out[row, chan, corr] = vis_in[row, chan, corr] / wsum
    else:
        def impl(vis_out, vis_in, row, chan, corr, weight_sum):
            pass

    return njit(nogil=True, cache=True)(impl)


def sigma_spectrum_normaliser_factory(present):
    if present:
        def impl(sigma_out, sigma_in, row, chan, corr, weight_sum):
            wsum = weight_sum[row, chan, corr]

            if wsum == 0.0:
                return

            # sqrt(sigma**2 * weight**2 / (weight(sum**2)))
            res = np.sqrt(sigma_in[row, chan, corr] / (wsum**2))
            sigma_out[row, chan, corr] = res
    else:
        def impl(sigma_out, sigma_in, row, chan, corr, weight_sum):
            pass

    return njit(nogil=True, cache=True)(impl)


def weight_spectrum_normaliser_factory(present):
    if present:
        def impl(wt_spec_out, wt_spec_in, row, chan, corr):
            wt_spec_out[row, chan, corr] = wt_spec_in[row, chan, corr]
    else:
        def impl(wt_spec_out, wt_spec_in, row, chan, corr):
            pass

    return njit(nogil=True, cache=True)(impl)


def chan_normaliser_factory(present):
    """ Returns function normalising channel data in a bin """
    if present:
        def impl(data_out, data_in, row, chan, corr, bin_size):
            data_out[row, chan, corr] = data_in[row, chan, corr] / bin_size
    else:
        def impl(data_out, data_in, row, chan, corr, bin_size):
            pass

    return njit(nogil=True, cache=True)(impl)


@generated_jit(nopython=True, nogil=True, cache=True)
def shape_or_invalid_shape(array, ndim):
    """ Return array shape tuple or (-1,)*ndim if the array is None """

    try:
        ndim_lit = getattr(ndim, "literal_value")
    except AttributeError:
        raise ValueError("ndim must be a integer literal")

    if is_numba_type_none(array):
        tup = (-1,)*ndim_lit

        def impl(array, ndim):
            return tup
    else:
        def impl(array, ndim):
            return array.shape

    return impl


@njit(nogil=True, cache=True)
def find_chan_corr(chan, corr, shape, chan_idx, corr_idx):
    """
    1. Get channel and correlation from shape if not set and the shape is valid
    2. Check they agree if they already agree

    Parameters
    ----------
    chan : int
        Existing channel size
    corr : int
        Existing correlation size
    shape : tuple
        Array shape tuple
    chan_idx : int
        Index of channel dimension in ``shape``.
    corr_idx : int
        Index of correlation dimension in ``shape``.

    Returns
    -------
    int
        Modified channel size
    int
        Modified correlation size
    """
    if chan_idx != -1:
        array_chan = shape[chan_idx]

        # Corresponds to a None array, ignore
        if array_chan == -1:
            pass
        # chan is not yet set, assign
        elif chan == 0:
            chan = array_chan
        # Check consistency
        elif chan != array_chan:
            raise ValueError("Inconsistent Channel Dimension "
                             "in Input Arrays")

    if corr_idx != -1:
        array_corr = shape[corr_idx]

        # Corresponds to a None array, ignore
        if array_corr == -1:
            pass
        # corr is not yet set, assign
        elif corr == 0:
            corr = array_corr
        # Check consistency
        elif corr != array_corr:
            raise ValueError("Inconsistent Correlation Dimension "
                             "in Input Arrays")

    return chan, corr


@njit(nogil=True, cache=True)
def chan_corrs(vis, flag,
               weight_spectrum, sigma_spectrum,
               chan_freq, chan_width,
               effective_bw, resolution):
    """
    Infer channel and correlation size from input dimensions

    Returns
    -------
    int
        channel size
    int
        correlation size
    """
    vis_shape = shape_or_invalid_shape(vis, 3)
    flag_shape = shape_or_invalid_shape(flag, 3)
    weight_spectrum_shape = shape_or_invalid_shape(weight_spectrum, 3)
    sigma_spectrum_shape = shape_or_invalid_shape(sigma_spectrum, 3)
    chan_freq_shape = shape_or_invalid_shape(chan_freq, 1)
    chan_width_shape = shape_or_invalid_shape(chan_width, 1)
    effective_bw_shape = shape_or_invalid_shape(effective_bw, 1)
    resolution_shape = shape_or_invalid_shape(resolution, 1)

    chan = 0
    corr = 0

    chan, corr = find_chan_corr(chan, corr, vis_shape, 1, 2)
    chan, corr = find_chan_corr(chan, corr, flag_shape, 1, 2)
    chan, corr = find_chan_corr(chan, corr, weight_spectrum_shape, 1, 2)
    chan, corr = find_chan_corr(chan, corr, sigma_spectrum_shape, 1, 2)
    chan, corr = find_chan_corr(chan, corr, chan_freq_shape, 0, -1)
    chan, corr = find_chan_corr(chan, corr, chan_width_shape, 0, -1)
    chan, corr = find_chan_corr(chan, corr, effective_bw_shape, 0, -1)
    chan, corr = find_chan_corr(chan, corr, resolution_shape, 0, -1)

    return chan, corr


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
def row_chan_average(row_meta, chan_meta, flag_row=None, weight=None,
                     vis=None, flag=None,
                     weight_spectrum=None, sigma_spectrum=None):

    have_flag_row = not is_numba_type_none(flag_row)
    have_vis = not is_numba_type_none(vis)
    have_flag = not is_numba_type_none(flag)
    have_weight = not is_numba_type_none(weight)
    have_weight_spectrum = not is_numba_type_none(weight_spectrum)
    have_sigma_spectrum = not is_numba_type_none(sigma_spectrum)

    flags_match = matching_flag_factory(have_flag_row)
    is_chan_flagged = is_chan_flagged_factory(have_flag)

    vis_factory = chan_output_factory(have_vis)
    weight_sum_factory = weight_sum_output_factory(have_vis)
    flag_factory = chan_output_factory(have_flag)
    weight_factory = chan_output_factory(have_weight_spectrum)
    sigma_factory = chan_output_factory(have_sigma_spectrum)

    vis_adder = vis_add_factory(have_vis,
                                have_weight,
                                have_weight_spectrum)
    weight_adder = chan_add_factory(have_weight_spectrum)
    sigma_adder = sigma_spectrum_add_factory(have_sigma_spectrum,
                                             have_weight,
                                             have_weight_spectrum)

    vis_normaliser = vis_normaliser_factory(have_vis)
    sigma_normaliser = sigma_spectrum_normaliser_factory(have_sigma_spectrum)
    weight_normaliser = weight_spectrum_normaliser_factory(
                            have_weight_spectrum)

    set_flagged = set_flagged_factory(have_flag)

    dummy_chan_freq = None
    dummy_chan_width = None

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

        vis_avg = vis_factory(out_shape, vis)
        vis_weight_sum = weight_sum_factory(out_shape, vis)
        weight_spectrum_avg = weight_factory(out_shape, weight_spectrum)
        sigma_spectrum_avg = sigma_factory(out_shape, sigma_spectrum)
        sigma_spectrum_weight_sum = sigma_factory(out_shape, sigma_spectrum)

        flagged_vis_avg = vis_factory(out_shape, vis)
        flagged_vis_weight_sum = weight_sum_factory(out_shape, vis)
        flagged_weight_spectrum_avg = weight_factory(out_shape,
                                                     weight_spectrum)
        flagged_sigma_spectrum_avg = sigma_factory(out_shape,
                                                   sigma_spectrum)
        flagged_sigma_spectrum_weight_sum = sigma_factory(out_shape,
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

                        vis_adder(flagged_vis_avg, flagged_vis_weight_sum, vis,
                                  weight, weight_spectrum,
                                  out_row, out_chan, in_row, in_chan, corr)
                        weight_adder(flagged_weight_spectrum_avg,
                                     weight_spectrum,
                                     out_row, out_chan, in_row, in_chan, corr)
                        sigma_adder(flagged_sigma_spectrum_avg,
                                    flagged_sigma_spectrum_weight_sum,
                                    sigma_spectrum,
                                    weight,
                                    weight_spectrum,
                                    out_row, out_chan, in_row, in_chan, corr)
                    else:
                        # Increment unflagged averages and counts
                        counts[out_row, out_chan, corr] += 1

                        vis_adder(vis_avg, vis_weight_sum, vis,
                                  weight, weight_spectrum,
                                  out_row, out_chan, in_row, in_chan, corr)
                        weight_adder(weight_spectrum_avg, weight_spectrum,
                                     out_row, out_chan, in_row, in_chan, corr)
                        sigma_adder(sigma_spectrum_avg,
                                    sigma_spectrum_weight_sum,
                                    sigma_spectrum,
                                    weight,
                                    weight_spectrum,
                                    out_row, out_chan, in_row, in_chan, corr)

        for r in range(out_rows):
            for f in range(out_chans):
                for c in range(ncorrs):
                    if counts[r, f, c] > 0:
                        # We have some unflagged samples and
                        # only these are used as averaged output
                        vis_normaliser(vis_avg, vis_avg,
                                       r, f, c,
                                       vis_weight_sum)
                        sigma_normaliser(sigma_spectrum_avg,
                                         sigma_spectrum_avg,
                                         r, f, c,
                                         sigma_spectrum_weight_sum)
                    elif flag_counts[r, f, c] > 0:
                        # We only have flagged samples and
                        # these are used as averaged output
                        vis_normaliser(vis_avg, flagged_vis_avg,
                                       r, f, c,
                                       flagged_vis_weight_sum)
                        sigma_normaliser(sigma_spectrum_avg,
                                         flagged_sigma_spectrum_avg,
                                         r, f, c,
                                         flagged_sigma_spectrum_weight_sum)
                        weight_normaliser(weight_spectrum_avg,
                                          flagged_weight_spectrum_avg,
                                          r, f, c)

                        # Flag the output bin
                        set_flagged(flag_avg, r, f, c)
                    else:
                        raise RowChannelAverageException("Zero-filled bin")

        return RowChanAverageOutput(vis_avg, flag_avg,
                                    weight_spectrum_avg,
                                    sigma_spectrum_avg)

    return impl


_chan_output_fields = ["chan_freq", "chan_width", "effective_bw", "resolution"]
ChannelAverageOutput = namedtuple("ChannelAverageOutput", _chan_output_fields)


@generated_jit(nopython=True, nogil=True, cache=True)
def chan_average(chan_meta, chan_freq=None, chan_width=None,
                 effective_bw=None, resolution=None):
    have_chan_freq = not is_numba_type_none(chan_freq)
    have_chan_width = not is_numba_type_none(chan_width)
    have_effective_bw = not is_numba_type_none(effective_bw)
    have_resolution = not is_numba_type_none(resolution)

    chan_freq_output = chan_output_factory(have_chan_freq)
    chan_width_output = chan_output_factory(have_chan_width)
    effective_bw_output = chan_output_factory(have_effective_bw)
    resolution_output = chan_output_factory(have_resolution)

    chan_freq_normaliser = normaliser_factory(have_chan_freq)

    chan_freq_adder = add_factory(have_chan_freq)
    chan_width_adder = add_factory(have_chan_width)
    effective_bw_adder = add_factory(have_effective_bw)
    resolution_adder = add_factory(have_resolution)

    def impl(chan_meta, chan_freq=None, chan_width=None,
             effective_bw=None, resolution=None):
        chan_map, out_chans = chan_meta

        chan_freq_avg = chan_freq_output(out_chans, chan_freq)
        chan_width_avg = chan_width_output(out_chans, chan_width)
        effective_bw_avg = effective_bw_output(out_chans, effective_bw)
        resolution_avg = resolution_output(out_chans, resolution)
        counts = np.zeros(out_chans, dtype=np.uint32)

        for in_chan, out_chan in enumerate(chan_map):
            counts[out_chan] += 1
            chan_freq_adder(chan_freq_avg, out_chan, chan_freq, in_chan)
            chan_width_adder(chan_width_avg, out_chan, chan_width, in_chan)
            effective_bw_adder(effective_bw_avg, out_chan,
                               effective_bw, in_chan)
            resolution_adder(resolution_avg, out_chan,
                             resolution, in_chan)

        for out_chan in range(out_chans):
            chan_freq_normaliser(chan_freq_avg, out_chan, counts[out_chan])

        return ChannelAverageOutput(chan_freq_avg, chan_width_avg,
                                    effective_bw_avg, resolution_avg)

    return impl


AverageOutput = namedtuple("AverageOutput",
                           ["time", "interval", "flag_row"] +
                           _row_output_fields +
                           _chan_output_fields +
                           _rowchan_output_fields)


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
