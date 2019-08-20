# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

from numba import types
import numpy as np

import sys
sys.path.insert(0, '/Users/smasoka/Varsity/codex-africanus/africanus/averaging/')
from support import unique_time, unique_baselines
from baseline_time_and_channel_mapping import baseline_row_mapper, baseline_chan_mapper

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
        def impl(data, row, bin_size):
            data[row] /= bin_size
    else:
        def impl(data, row, bin_size):
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

def matching_flag_factory(present):
    if present:
        def impl(flag_row, ri, out_flag_row, ro):
            return flag_row[ri] == out_flag_row[ro]
    else:
        def impl(flag_row, ri, out_flag_row, ro):
            return True

    return njit(nogil=True, cache=True)(impl)


_row_output_fields = ["interval", "antenna1", "antenna2", "time_centroid", "exposure",
                      "uvw", "weight", "sigma"]
RowAverageOutput = namedtuple("RowAverageOutput", _row_output_fields)


@generated_jit(nopython=True, nogil=True, cache=True)
def baseline_row_average(meta, ant1, ant2, interval=None, flag_row=None, time_centroid=None, 
                         exposure=None, uvw=None, weight=None, sigma=None):
    
    # Verify Data Types
    have_interval = not is_numba_type_none(interval)
    have_flag_row = not is_numba_type_none(flag_row)
    have_uvw = not is_numba_type_none(uvw)
    have_time_centroid = not is_numba_type_none(time_centroid)
    have_exposure = not is_numba_type_none(exposure)
    have_weight = not is_numba_type_none(weight)
    have_sigma = not is_numba_type_none(sigma)
    
    # Make sure about the flags
    flags_match = matching_flag_factory(have_flag_row)
    
    # Factory functions 
    interval_factory = output_factory(have_interval)
    uvw_factory = output_factory(have_uvw)
    time_centroid_factory = output_factory(have_time_centroid)
    exposure_factory = output_factory(have_exposure)
    weight_factory = output_factory(have_weight)
    sigma_factory = output_factory(have_sigma)
    
    # Factory
    interval_adder = add_factory(have_interval)
    time_centroid_adder = add_factory(have_time_centroid)
    exposure_adder = add_factory(have_exposure)
    
    uvw_adder = comp_add_factory(have_uvw)
    weight_adder = comp_add_factory(have_weight)
    sigma_adder = comp_add_factory(have_sigma)
    
    # Normalise
    uvw_normaliser = normaliser_factory(have_uvw)
    time_centroid_normaliser = normaliser_factory(have_time_centroid)
    weight_normaliser = normaliser_factory(have_weight)
    sigma_normaliser = normaliser_factory(have_sigma)
    
    def impl(meta, ant1, ant2, interval=None, flag_row=None, time_centroid=None, 
            exposure=None, uvw=None, weight=None, sigma=None):
        
        out_rows = meta.time.shape[0]
        print(out_rows)
        print(uvw.shape)
        counts = np.zeros(out_rows, dtype=np.uint32)
        ant1_avg = np.empty(out_rows, ant1.dtype)
        ant2_avg = np.empty(out_rows, ant2.dtype)
        
        interval_avg = interval_factory(out_rows, interval)
        uvw_avg = uvw_factory(out_rows, uvw)
        time_centroid_avg = time_centroid_factory(out_rows, time_centroid)
        exposure_avg = exposure_factory(out_rows, exposure)
        weight_avg = weight_factory(out_rows, weight)
        sigma_avg = sigma_factory(out_rows, sigma)
        
        # 
        for in_row, out_row in enumerate(meta.map):
            if flags_match(flag_row, in_row, meta.flag_row, out_row):
                interval_adder(interval_avg, out_row, interval, in_row)
                uvw_adder(uvw_avg, out_row, uvw, in_row)
                weight_adder(weight_avg, out_row, weight, in_row)
                sigma_adder(sigma_avg, out_row, sigma, in_row)
                time_centroid_adder(time_centroid_avg, out_row, time_centroid, in_row)
                exposure_adder(exposure_avg, out_row, exposure, in_row)
                
                counts[out_row] += 1
            
            ant1_avg[out_row] = ant1[in_row]
            ant2_avg[out_row] = ant2[in_row]
        
        
        # Normalise
        for out_row in range(out_rows):
            count = counts[out_row]
            
            if count > 0:
                uvw_normaliser(uvw_avg, out_row, count)
                weight_normaliser(weight_avg, out_row, count)
                sigma_normaliser(sigma_avg, out_row, count)
                time_centroid_normaliser(time_centroid_avg, out_row, count)
        
        return RowAverageOutput(interval_avg, ant1_avg, ant2_avg,
                                time_centroid_avg, exposure_avg, uvw_avg,
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
def baseline_row_chan_average(row_meta, chan_meta,flag_row=None, vis=None, flag=None, 
                              weight_spectrum=None, sigma_spectrum=None, baseline_chan_bin_size=1):
    
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
    
    
    def impl(row_meta, chan_meta,flag_row=None, vis=None, flag=None,
             weight_spectrum=None, sigma_spectrum=None, baseline_chan_bin_size=1):
        
        out_rows = row_meta.time.shape[0]
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
                
        return RowChanAverageOutput(vis_avg, flag_avg, weight_spectrum_avg, sigma_spectrum_avg)
    
    return impl



AverageOutput = namedtuple("AverageOutput",
                           ["time", "flag_row"] + _row_output_fields 
                           + _rowchan_output_fields)

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
def baseline_time_and_channel(time, interval, antenna1, antenna2,
                     time_centroid=None, exposure=None, flag_row=None,
                     uvw=None, weight=None, sigma=None,
                     vis=None, flag=None,
                     weight_spectrum=None, sigma_spectrum=None,
                     bins_for_longest_baseline=1.0, baseline_chan_bin_size=1):
    
    # Verify data types
    valid_types = (types.misc.Omitted, types.scalars.Float,
                   types.scalars.Integer)

    if not isinstance(bins_for_longest_baseline, valid_types):
        raise TypeError("bins_for_longest_baseline must be a scalar float")

    valid_types = (types.misc.Omitted, types.scalars.Integer)

    if not isinstance(baseline_chan_bin_size, valid_types):
        raise TypeError("chan_bin_size must be a scalar integer")
    
    
    have_vis = not is_numba_type_none(vis)
    have_flag = not is_numba_type_none(flag)
    have_weight = not is_numba_type_none(weight_spectrum)
    have_sigma = not is_numba_type_none(sigma_spectrum)
    
    chan_corrs = chan_corr_factory(have_vis, have_flag, have_weight, have_sigma)
    
    def impl(time, interval, antenna1, antenna2,
                     time_centroid=None, exposure=None, flag_row=None,
                     uvw=None, weight=None, sigma=None,
                     vis=None, flag=None,
                     weight_spectrum=None, sigma_spectrum=None,
                     bins_for_longest_baseline=1.0, baseline_chan_bin_size=1.0):
        
        print("In baseline time and channel averaging")
        
        nchan, ncorr = chan_corrs(vis, flag, weight_spectrum, sigma_spectrum)
        
        flag_row = merge_flags(flag_row, flag)
    
        # Get the baseline row mapper data
        row_meta = baseline_row_mapper(uvw, time, antenna1, antenna2, flag_row=flag_row,
                                       bins_for_longest_baseline=bins_for_longest_baseline)
    
        chan_meta = baseline_chan_mapper(uvw, antenna1, antenna2, nchan,
                                         baseline_chan_bin_size=baseline_chan_bin_size)
        
        # Average the rows according to the meta data
        row_data = baseline_row_average(row_meta, antenna1, antenna2, interval, flag_row=flag_row,
                                        time_centroid=time_centroid, exposure=exposure, uvw=uvw,
                                        weight=weight, sigma=sigma)
        
        chan_data = baseline_chan_average(row_meta, chan_meta, flag_row=flag_row, vis=vis, 
                                          flag=flag, weight_spectrum=weight_spectrum, sigma_spectrum=sigma_spectrum,
                                          baseline_chan_bin_size=baseline_chan_bin_size)
        
        return AverageOutput(row_meta.time,
                             row_meta.flag_row, 
                             row_data.interval, 
                             row_data.antenna1, 
                             row_data.antenna2,
                             row_data.time_centroid, 
                             row_data.exposure, 
                             row_data.uvw, 
                             row_data.weight, 
                             row_data.sigma,
                             chan_data.vis,
                             chan_data.flag,
                             chan_data.weight_spectrum,
                             chan_data.sigma_spectrum)
    
    return impl
