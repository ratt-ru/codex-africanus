# -*- coding: utf-8 -*-

# flake8: noqa

from collections import namedtuple

import numpy as np

from africanus.averaging.shared import (chan_corrs,
                                        flags_match)
from africanus.util.numba import (generated_jit,
                                  overload,
                                  njit,
                                  is_numba_type_none)


_row_output_fields = ["antenna1", "antenna2", "time_centroid", "exposure",
                      "uvw", "weight", "sigma"]
RowAverageOutput = namedtuple("RowAverageOutput", _row_output_fields)


@generated_jit(nopython=True, nogil=True, cache=True)
def row_average(meta, ant1, ant2, flag_row=None,
                time_centroid=None, exposure=None, uvw=None,
                weight=None, sigma=None):

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
        for in_row in range(meta.map.shape[0]):
            for c in range(meta.map.shape[1]):
                out_row = meta.map[in_row, c]
                # Here we can simply assign because input_row baselines
                # should always match output row baselines
                ant1_avg[out_row] = ant1[in_row]
                ant2_avg[out_row] = ant2[in_row]

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


@njit(nogil=True, inline='always')
def chan_add(output, input, orow, irow, ichan, corr):
    if input is not None:
        output[orow, corr] += input[irow, ichan, corr]


def vis_add(out_vis, out_weight_sum, in_vis,
             weight, weight_spectrum,
             orow, irow, ichan, corr):
    pass


@overload(vis_add, inline='always')
def _vis_add(out_vis, out_weight_sum, in_vis,
             weight, weight_spectrum,
             orow, irow, ichan, corr):
    """ Returns function adding weighted visibilities to a bin """
    if is_numba_type_none(in_vis):
        def impl(out_vis, out_weight_sum, in_vis,
                 weight, weight_spectrum,
                 orow, irow, ichan, corr):

            pass
    elif not is_numba_type_none(weight_spectrum):
        # Always prefer more accurate weight spectrum if we have it

        def impl(out_vis, out_weight_sum, in_vis,
                 weight, weight_spectrum,
                 orow, irow, ichan, corr):

            wt = weight_spectrum[irow, ichan, corr]
            iv = in_vis[irow, ichan, corr] * wt
            out_vis[orow, corr] += iv
            out_weight_sum[orow, corr] += wt
    elif not is_numba_type_none(weight):
        # Otherwise fall back to row weights
        def impl(out_vis, out_weight_sum, in_vis,
                 weight, weight_spectrum,
                 orow, irow, ichan, corr):

            wt = weight[irow, corr]
            iv = in_vis[irow, ichan, corr] * wt
            out_vis[orow, corr] += iv
            out_weight_sum[orow, corr] += wt
    else:
        # Natural weights
        def impl(out_vis, out_weight_sum, in_vis,
                 weight, weight_spectrum,
                 orow, irow, ichan, corr):

            iv = in_vis[irow, ichan, corr]
            out_vis[orow, corr] += iv
            out_weight_sum[orow, corr] += 1.0

    return impl


def sigma_spectrum_add(out_sigma, out_weight_sum, in_sigma,
                       weight, weight_spectrum,
                       orow, irow, ichan, corr):
    pass


@overload(sigma_spectrum_add, inline="always")
def _sigma_spectrum_add(out_sigma, out_weight_sum, in_sigma,
                        weight, weight_spectrum,
                        orow, irow, ichan, corr):
    """ Returns function adding weighted sigma to a bin """
    if is_numba_type_none(in_sigma):
        def impl(out_sigma, out_weight_sum, in_sigma,
                 weight, weight_spectrum,
                 orow, irow, ichan, corr):
            pass
    elif not is_numba_type_none(weight_spectrum):
        def impl(out_sigma, out_weight_sum, in_sigma,
                 weight, weight_spectrum,
                 orow, irow, ichan, corr):

            # Always prefer more accurate weight spectrum if we have it
            # sum(sigma**2 * weight**2)
            wt = weight_spectrum[irow, ichan, corr]
            is_ = in_sigma[irow, ichan, corr]**2 * wt**2
            out_sigma[orow, corr] += is_
            out_weight_sum[orow, corr] += wt

    elif not is_numba_type_none(weight):
        def impl(out_sigma, out_weight_sum, in_sigma,
                 weight, weight_spectrum,
                 orow, irow, ichan, corr):

            # sum(sigma**2 * weight**2)
            wt = weight[irow, corr]
            is_ = in_sigma[irow, ichan, corr]**2 * wt**2
            out_sigma[orow, corr] += is_
            out_weight_sum[orow, corr] += wt
    else:
        # Natural weights
        # sum(sigma**2 * weight**2)

        def impl(out_sigma, out_weight_sum, in_sigma,
                 weight, weight_spectrum,
                 orow, irow, ichan, corr):

            out_sigma[orow, corr] += in_sigma[irow, ichan, corr]**2
            out_weight_sum[orow, corr] += 1.0

    return impl


def normalise_vis(vis_out, vis_in, row, corr, weight_sum):
    pass


@overload(normalise_vis, inline='always')
def _normalise_vis(vis_out, vis_in, row, corr, weight_sum):
    if is_numba_type_none(vis_in):
        def impl(vis_out, vis_in, row, corr, weight_sum):
            pass
    else:
        def impl(vis_out, vis_in, row, corr, weight_sum):
            wsum = weight_sum[row, corr]

            if wsum != 0.0:
                vis_out[row, corr] = vis_in[row, corr] / wsum
    return impl


def normalise_sigma_spectrum(sigma_out, sigma_in, row, corr, weight_sum):
    pass


@overload(normalise_sigma_spectrum, inline='always')
def _normalise_sigma_spectrum(sigma_out, sigma_in,
                              row, corr,
                              weight_sum):
    if is_numba_type_none(sigma_in) or is_numba_type_none(weight_sum):
        def impl(sigma_out, sigma_in, row, corr, weight_sum):
            pass
    else:
        def impl(sigma_out, sigma_in, row, corr, weight_sum):
            wsum = weight_sum[row, corr]

            if wsum == 0.0:
                return

            # sqrt(sigma**2 * weight**2 / (weight(sum**2)))
            res = np.sqrt(sigma_in[row, corr] / (wsum**2))
            sigma_out[row, corr] = res

    return impl


def normalise_weight_spectrum(wt_spec_out, wt_spec_in, row, corr):
    pass


@overload(normalise_weight_spectrum, inline='always')
def _normalise_weight_spectrum(wt_spec_out, wt_spec_in, row, corr):
    if is_numba_type_none(wt_spec_in):
        def impl(wt_spec_out, wt_spec_in, row, corr):
            pass
    else:
        def impl(wt_spec_out, wt_spec_in, row, corr):
            wt_spec_out[row, corr] = wt_spec_in[row, corr]

    return impl


_rowchan_output_fields = ["vis", "flag", "weight_spectrum", "sigma_spectrum"]
RowChanAverageOutput = namedtuple("RowChanAverageOutput",
                                  _rowchan_output_fields)


class RowChannelAverageException(Exception):
    pass


@generated_jit(nopython=True, nogil=True, cache=True)
def row_chan_average(meta, flag_row=None, weight=None,
                     vis=None, flag=None,
                     weight_spectrum=None, sigma_spectrum=None):

    dummy_chan_freq = None
    dummy_chan_width = None

    weight_add = chan_add
    sigma_add = sigma_spectrum_add
    normalise_weights = normalise_weight_spectrum

    def impl(meta, flag_row=None, weight=None,
             vis=None, flag=None,
             weight_spectrum=None, sigma_spectrum=None):

        out_rows = meta.time.shape[0]
        nchan, ncorrs = chan_corrs(vis, flag,
                                   weight_spectrum, sigma_spectrum,
                                   dummy_chan_freq, dummy_chan_width,
                                   dummy_chan_width, dummy_chan_width)

        out_shape = (out_rows, ncorrs)

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

        if weight_spectrum is None:
            weight_spectrum_avg = None
            flagged_weight_spectrum_avg = None
        else:
            weight_spectrum_avg = np.zeros(
                out_shape, dtype=weight_spectrum.dtype)
            flagged_weight_spectrum_avg = np.zeros_like(weight_spectrum_avg)

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

        if flag is None:
            flag_avg = None
        else:
            flag_avg = np.zeros(out_shape, dtype=flag.dtype)

        counts = np.zeros(out_shape, dtype=np.uint32)
        flag_counts = np.zeros(out_shape, dtype=np.uint32)

        # Iterate over input rows, accumulating into output rows
        for in_row in range(meta.map.shape[0]):
            for in_chan in range(meta.map.shape[1]):
                out_row = meta.map[in_row, in_chan]

                # TIME_CENTROID/EXPOSURE case applies here,
                # must have flagged input and output OR unflagged input and output
                if not flags_match(flag_row, in_row, meta.flag_row, out_row):
                    continue

                for corr in range(ncorrs):
                    if flag is not None and flag[in_row, in_chan, corr] != 0:
                        # Increment flagged averages and counts
                        flag_counts[out_row, corr] += 1

                        vis_add(flagged_vis_avg, flagged_vis_weight_sum, vis,
                                weight, weight_spectrum,
                                out_row, in_row, in_chan, corr)
                        weight_add(flagged_weight_spectrum_avg,
                                   weight_spectrum,
                                   out_row, in_row, in_chan, corr)
                        sigma_add(flagged_sigma_spectrum_avg,
                                  flagged_sigma_spectrum_weight_sum,
                                  sigma_spectrum,
                                  weight,
                                  weight_spectrum,
                                  out_row, in_row, in_chan, corr)
                    else:
                        # Increment unflagged averages and counts
                        counts[out_row, corr] += 1

                        vis_add(vis_avg, vis_weight_sum, vis,
                                weight, weight_spectrum,
                                out_row, in_row, in_chan, corr)
                        weight_add(weight_spectrum_avg, weight_spectrum,
                                   out_row, in_row, in_chan, corr)
                        sigma_add(sigma_spectrum_avg,
                                  sigma_spectrum_weight_sum,
                                  sigma_spectrum,
                                  weight,
                                  weight_spectrum,
                                  out_row, in_row, in_chan, corr)

        for r in range(out_rows):
            for c in range(ncorrs):
                if counts[r, c] > 0:
                    # We have some unflagged samples and
                    # only these are used as averaged output
                    normalise_vis(vis_avg, vis_avg,
                                    r, c,
                                    vis_weight_sum)
                    normalise_sigma_spectrum(sigma_spectrum_avg,
                                                sigma_spectrum_avg,
                                                r, c,
                                                sigma_spectrum_weight_sum)
                elif flag_counts[r, c] > 0:
                    # We only have flagged samples and
                    # these are used as averaged output
                    normalise_vis(
                            vis_avg, flagged_vis_avg,
                            r, c,
                            flagged_vis_weight_sum)
                    normalise_sigma_spectrum(
                            sigma_spectrum_avg,
                            flagged_sigma_spectrum_avg,
                            r, c,
                            flagged_sigma_spectrum_weight_sum)
                    normalise_weights(
                            weight_spectrum_avg,
                            flagged_weight_spectrum_avg,
                            r, c)

                    # Flag the output bin
                    if flag_avg is not None:
                        flag_avg[r, c] = 1
                else:
                    raise RowChannelAverageException("Zero-filled bin")




        return RowChanAverageOutput(vis_avg, flag_avg,
                                    weight_spectrum_avg,
                                    sigma_spectrum_avg)

    return impl
