# -*- coding: utf-8 -*-

# flake8: noqa

from collections import namedtuple

import numpy as np

from africanus.averaging.bda_mapping import atemkeng_mapper
from africanus.averaging.shared import (chan_corrs,
                                        flags_match,
                                        merge_flags)
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

    have_flag_row = not is_numba_type_none(flag_row)
    have_time_centroid = not is_numba_type_none(time_centroid)
    have_exposure = not is_numba_type_none(exposure)
    have_uvw = not is_numba_type_none(uvw)
    have_weight = not is_numba_type_none(weight)
    have_sigma = not is_numba_type_none(sigma)

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
            None if not have_uvw else
            np.zeros((out_rows,) + uvw.shape[1:],
                     dtype=uvw.dtype))

        time_centroid_avg = (
            None if not have_time_centroid else
            np.zeros((out_rows,) + time_centroid.shape[1:],
                     dtype=time_centroid.dtype))

        exposure_avg = (
            None if not have_exposure else
            np.zeros((out_rows,) + exposure.shape[1:],
                     dtype=exposure.dtype))

        weight_avg = (
            None if not have_weight else
            np.zeros((out_rows,) + weight.shape[1:],
                     dtype=weight.dtype))

        sigma_avg = (
            None if not have_sigma else
            np.zeros((out_rows,) + sigma.shape[1:],
                     dtype=sigma.dtype))

        sigma_weight_sum = (
            None if not have_sigma else
            np.zeros((out_rows,) + sigma.shape[1:],
                     dtype=sigma.dtype))

        # Iterate over input rows, accumulating into output rows
        for ri in range(meta.map.shape[0]):
            for fi in range(meta.map.shape[1]):
                ro = meta.map[ri, fi]
                # Here we can simply assign because input_row baselines
                # should always match output row baselines
                ant1_avg[ro] = ant1[ri]
                ant2_avg[ro] = ant2[ri]

                # Input and output flags must match in order for the
                # current row to contribute to these columns
                if have_flag_row and flag_row[ri] != meta.flag_row[ro]:
                    continue

                if have_uvw:
                    uvw_avg[ro, 0] += uvw[ri, 0]
                    uvw_avg[ro, 1] += uvw[ri, 1]
                    uvw_avg[ro, 2] += uvw[ri, 2]

                if have_time_centroid:
                    time_centroid_avg[ro] += time_centroid[ri]

                if have_exposure:
                    exposure_avg[ro] += exposure[ri]

                if have_weight:
                    for co in range(weight.shape[1]):
                        weight_avg[ro, co] += weight[ri, co]

                if have_sigma:
                    for co in range(sigma.shape[1]):
                        # Use weights if present else natural weights
                        wt = weight[ri, co] if have_weight else 1.0

                        # Assign
                        sigma_avg[ro, co] += sigma[ri, co]**2 * wt**2
                        sigma_weight_sum[ro, co] += wt

                counts[ro] += 1

        # Normalise
        for ro in range(out_rows):
            count = counts[ro]

            if count > 0:
                # Normalise uvw
                if have_uvw:
                    uvw_avg[ro, 0] /= count
                    uvw_avg[ro, 1] /= count
                    uvw_avg[ro, 2] /= count

                # Normalise time centroid
                if have_time_centroid:
                    time_centroid_avg[ro] /= count

                # Normalise sigma
                if have_sigma:
                    for co in range(sigma.shape[1]):
                        ssva = sigma_avg[ro, co]
                        wt = sigma_weight_sum[ro, co]

                        if wt != 0.0:
                            ssva /= (wt**2)

                        sigma_avg[ro, co] = np.sqrt(ssva)

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

    have_flag_row = not is_numba_type_none(flag_row)
    have_weight = not is_numba_type_none(weight)

    have_vis = not is_numba_type_none(vis)
    have_flag = not is_numba_type_none(flag)
    have_weight_spectrum = not is_numba_type_none(weight_spectrum)
    have_sigma_spectrum = not is_numba_type_none(sigma_spectrum)

    def impl(meta, flag_row=None, weight=None,
             vis=None, flag=None,
             weight_spectrum=None, sigma_spectrum=None):

        out_rows = meta.time.shape[0]
        nchan, ncorrs = chan_corrs(vis, flag,
                                   weight_spectrum, sigma_spectrum,
                                   dummy_chan_freq, dummy_chan_width,
                                   dummy_chan_width, dummy_chan_width)

        out_shape = (out_rows, ncorrs)

        if have_vis:
            vis_avg = np.zeros(out_shape, dtype=vis.dtype)
            vis_weight_sum = np.zeros(out_shape, dtype=vis.real.dtype)
            flagged_vis_avg = np.zeros_like(vis_avg)
            flagged_vis_weight_sum = np.zeros_like(vis_weight_sum)
        else:
            vis_avg = None
            vis_weight_sum = None
            flagged_vis_avg = None
            flagged_vis_weight_sum = None

        if have_weight_spectrum:
            weight_spectrum_avg = np.zeros(
                out_shape, dtype=weight_spectrum.dtype)
            flagged_weight_spectrum_avg = np.zeros_like(weight_spectrum_avg)
        else:
            weight_spectrum_avg = None
            flagged_weight_spectrum_avg = None

        if have_sigma_spectrum:
            sigma_spectrum_avg = np.zeros(
                out_shape, dtype=sigma_spectrum.dtype)
            sigma_spectrum_weight_sum = np.zeros_like(sigma_spectrum_avg)
            flagged_sigma_spectrum_avg = np.zeros_like(sigma_spectrum_avg)
            flagged_sigma_spectrum_weight_sum = np.zeros_like(
                sigma_spectrum_avg)
        else:
            sigma_spectrum_avg = None
            sigma_spectrum_weight_sum = None
            flagged_sigma_spectrum_avg = None
            flagged_sigma_spectrum_weight_sum = None

        if have_flag:
            flag_avg = np.zeros(out_shape, dtype=flag.dtype)
        else:
            flag_avg = None

        counts = np.zeros(out_shape, dtype=np.uint32)
        flag_counts = np.zeros(out_shape, dtype=np.uint32)

        # Iterate over input rows, accumulating into output rows
        for ri in range(meta.map.shape[0]):
            for fi in range(meta.map.shape[1]):
                ro = meta.map[ri, fi]

                # TIME_CENTROID/EXPOSURE case applies here,
                # must have flagged input and output OR unflagged input and output
                if have_flag_row and flag_row[ri] != meta.flag_row[ro]:
                    continue

                for co in range(ncorrs):
                    flagged = have_flag and flag[ri, fi, co] != 0

                    if flagged:
                        flag_counts[ro, co] += 1
                    else:
                        counts[ro, co] += 1

                    # Aggregate visibilities
                    if have_vis:
                        # Use full-resolution weight spectrum if given
                        # else weights, else natural weights
                        wt = (weight_spectrum[ri, fi, co]
                              if have_weight_spectrum else
                              weight[ri, co] if have_weight else 1.0)

                        iv = vis[ri, fi, co] * wt

                        if flagged:
                            flagged_vis_avg[ro, co] += iv
                            flagged_vis_weight_sum[ro, co] += wt
                        else:
                            vis_avg[ro, co] += iv
                            vis_weight_sum[ro, co] += wt


                    # Weight Spectrum
                    if have_weight_spectrum:
                        if flagged:
                            flagged_weight_spectrum_avg[ro, co] += (
                                weight_spectrum[ri, fi, co])
                        else:
                            weight_spectrum_avg[ro, co] += (
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
                            flagged_sigma_spectrum_avg[ro, co] += ssin
                            flagged_sigma_spectrum_weight_sum[ro, co] += wt
                        else:
                            sigma_spectrum_avg[ro, co] += ssin
                            sigma_spectrum_weight_sum[ro, co] += wt

        for ro in range(out_rows):
            for co in range(ncorrs):
                if counts[ro, co] > 0:
                    if have_vis:
                        vwsum = vis_weight_sum[ro, co]
                        vin = vis_avg[ro, co]

                    if have_sigma_spectrum:
                        sswsum = sigma_spectrum_weight_sum[ro, co]
                        ssin = sigma_spectrum_avg[ro, co]

                    flagged = 0
                elif flag_counts[ro, co] > 0:
                    if have_vis:
                        vwsum = flagged_vis_weight_sum[ro, co]
                        vin = flagged_vis_avg[ro, co]

                    if have_sigma_spectrum:
                        sswsum = flagged_sigma_spectrum_weight_sum[ro, co]
                        ssin = flagged_sigma_spectrum_avg[ro, co]

                    flagged = 1
                else:
                    raise RowChannelAverageException("Zero-filled bin")

                # Normalise visibilities
                if have_vis and vwsum != 0.0:
                    vis_avg[ro, co] = vin / vwsum

                # Normalise Sigma Spectrum
                if have_sigma_spectrum and sswsum != 0.0:
                    # sqrt(sigma**2 * weight**2 / (weight(sum**2)))
                    sigma_spectrum_avg[ro, co] = np.sqrt(ssin / sswsum**2)

                # Set flag
                if have_flag:
                    flag_avg[ro, co] = flagged

                # Copy Weights if flagged
                if have_weight_spectrum and flagged:
                    weight_spectrum_avg[ro, co] = (
                        flagged_weight_spectrum_avg[ro, co])

        return RowChanAverageOutput(vis_avg, flag_avg,
                                    weight_spectrum_avg,
                                    sigma_spectrum_avg)

    return impl


_chan_output_fields = ["chan_freq", "chan_width", "effective_bw", "resolution"]
ChannelAverageOutput = namedtuple("ChannelAverageOutput", _chan_output_fields)



AverageOutput = namedtuple("AverageOutput",
                           ["time", "interval", "flag_row"] +
                           _row_output_fields +
                           _chan_output_fields +
                           _rowchan_output_fields)


@generated_jit(nopython=True, nogil=True, cache=True)
def bda(time, interval, antenna1, antenna2, ref_freq,
        time_centroid=None, exposure=None, flag_row=None,
        uvw=None, weight=None, sigma=None,
        chan_freq=None, chan_width=None,
        effective_bw=None, resolution=None,
        vis=None, flag=None,
        weight_spectrum=None, sigma_spectrum=None,
        max_uvw_dist=None, lm_max=1.0,
        decorrelation=0.98):

    def impl(time, interval, antenna1, antenna2, ref_freq,
             time_centroid=None, exposure=None, flag_row=None,
             uvw=None, weight=None, sigma=None,
             chan_freq=None, chan_width=None,
             effective_bw=None, resolution=None,
             vis=None, flag=None,
             weight_spectrum=None, sigma_spectrum=None,
             max_uvw_dist=None, lm_max=1.0,
             decorrelation=0.98):

        # Merge flag_row and flag arrays
        flag_row = merge_flags(flag_row, flag)

        meta = atemkeng_mapper(time, interval, antenna1, antenna2, uvw,
                               ref_freq, max_uvw_dist, chan_width,
                               flag_row=flag_row,
                               lm_max=lm_max,
                               decorrelation=decorrelation)

        row_avg = row_average(meta, antenna1, antenna2, flag_row,  # noqa: F841
                              time_centroid, exposure, uvw,
                              weight=weight, sigma=sigma)

        row_chan_avg = row_chan_average(meta,  # noqa: F841
                                        flag_row=flag_row,
                                        vis=vis, flag=flag,
                                        weight_spectrum=weight_spectrum,
                                        sigma_spectrum=sigma_spectrum)


        # Have to explicitly write it out because numba tuples
        # are highly constrained types
        return AverageOutput(meta.time,
                             meta.interval,
                             meta.flag_row,
                             row_avg.antenna1,
                             row_avg.antenna2,
                             row_avg.time_centroid,
                             row_avg.exposure,
                             row_avg.uvw,
                             row_avg.weight,
                             row_avg.sigma,
                            #  chan_data.chan_freq,
                            #  chan_data.chan_width,
                            #  chan_data.effective_bw,
                            #  chan_data.resolution,
                             None,
                             None,
                             None,
                             None,
                             row_chan_avg.vis,
                             row_chan_avg.flag,
                             row_chan_avg.weight_spectrum,
                             row_chan_avg.sigma_spectrum)


    return impl
