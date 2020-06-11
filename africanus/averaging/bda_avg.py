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
                    # Select output arrays depending on whether the bin is flagged
                    f = flag is not None and flag[in_row, in_chan, corr] != 0
                    cnt = flag_counts if f else counts
                    ovis = flagged_vis_avg if f else vis_avg
                    ovis_ws = flagged_vis_weight_sum if f else vis_weight_sum
                    ows = (flagged_weight_spectrum_avg if f
                           else weight_spectrum_avg)
                    oss = (flagged_sigma_spectrum_avg if f
                           else sigma_spectrum_avg)
                    oss_ws = (flagged_sigma_spectrum_weight_sum if f
                              else sigma_spectrum_weight_sum)

                    cnt[out_row, corr] += 1

                    # Aggregate visibilities
                    if vis is None:
                        pass
                    elif weight_spectrum is not None:
                        # Use full-resolution weight spectrum if given
                        wt = weight_spectrum[in_row, in_chan, corr]
                        iv = vis[in_row, in_chan, corr] * wt
                        ovis[out_row, corr] += iv
                        ovis_ws[out_row, corr] += wt
                    elif weight is not None:
                        # Otherwise use weight column
                        wt = weight[in_row, corr]
                        iv = vis[in_row, in_chan, corr]
                        ovis[out_row, corr] += iv
                        ovis_ws[out_row, corr] += wt
                    else:
                        # Otherwise use natural weights
                        iv = vis[in_row, in_chan, corr]
                        ovis[out_row, corr] += iv
                        ovis_ws[out_row, corr] += 1.0

                    # Weight Spectrum
                    if weight_spectrum is not None:
                        ows[out_row, corr] += weight_spectrum[in_row, in_chan, corr]

                    # Sigma Spectrum
                    if sigma_spectrum is None:
                        pass
                    elif weight_spectrum is not None:
                        # Use full-resolution weight spectrum if given
                        wt = weight_spectrum[in_row, in_chan, corr]
                        is_ = sigma_spectrum[in_row, in_chan, corr]**2 * wt**2
                        oss[out_row, corr] += is_
                        oss_ws[out_row, corr] += wt
                    elif weight is not None:
                        # Otherwise use weight column
                        # sum(sigma**2 * weight**2)
                        wt = weight[in_row, corr]
                        is_ = sigma_spectrum[in_row, in_chan, corr]**2 * wt**2
                        oss[out_row, corr] += is_
                        oss_ws[out_row, corr] += wt
                    else:
                        # Otherwise use natural weights
                        wt = 1.0
                        is_ = sigma_spectrum[in_row, in_chan, corr]**2 * wt**2
                        oss[out_row, corr] += is_
                        oss_ws[out_row, corr] += wt

        for r in range(out_rows):
            for c in range(ncorrs):
                if counts[r, c] > 0:
                    # Normalise Visibilities
                    vwsum = vis_weight_sum[r, corr]

                    if vwsum != 0.0:
                        vis_avg[r, c] = vis_avg[r, c] / vwsum

                    # Normalise Sigma Spectrum
                    sswsum = sigma_spectrum_weight_sum[r, corr]

                    if sswsum != 0.0:
                        # sqrt(sigma**2 * weight**2 / (weight(sum**2)))
                        res = sigma_spectrum_avg[r, corr] / (sswsum**2)
                        sigma_spectrum_avg[r, corr] = np.sqrt(res)

                    # Unflag the output bin
                    if flag_avg is not None:
                        flag_avg[r, c] = 0

                elif flag_counts[r, c] > 0:
                    # We only have flagged samples and
                    # these are used as averaged output

                    # Normalise Visibilities
                    vwsum = flagged_vis_weight_sum[r, corr]

                    if vwsum != 0.0:
                        vis_avg[r, c] = flagged_vis_avg[r, c] / vwsum

                    # Normalise Sigma Spectrum
                    sswsum = flagged_sigma_spectrum_weight_sum[r, corr]

                    if sswsum != 0.0:
                        # sqrt(sigma**2 * weight**2 / (weight(sum**2)))
                        fss = flagged_sigma_spectrum_avg[r, corr]
                        sigma_spectrum_avg[r, corr] = np.sqrt(fss / sswsum**2)

                    # Copy Weights
                    weight_spectrum_avg[r, c] = flagged_weight_spectrum_avg[r, c]

                    # Flag the output bin
                    if flag_avg is not None:
                        flag_avg[r, c] = 1
                else:
                    raise RowChannelAverageException("Zero-filled bin")


        return RowChanAverageOutput(vis_avg, flag_avg,
                                    weight_spectrum_avg,
                                    sigma_spectrum_avg)

    return impl
