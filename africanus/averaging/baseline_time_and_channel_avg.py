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
from baseline_time_and_channel_mapping import baseline_row_mapper

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

_row_output_fields = ["antenna1", "antenna2", "time_centroid", "exposure",
                      "uvw", "weight", "sigma"]
RowAverageOutput = namedtuple("RowAverageOutput", _row_output_fields)


@generated_jit(nopython=True, nogil=True, cache=True)
def baseline_row_average(meta, ant1, ant2, flag_row=None,time_centroid=None, 
                         exposure=None, uvw=None, weight=None, sigma=None):
    
    # Verify Data Types
    have_flag_row = not is_numba_type_none(flag_row)
    have_uvw = not is_numba_type_none(uvw)
    have_time_centroid = not is_numba_type_none(time_centroid)
    have_exposure = not is_numba_type_none(exposure)
    have_weight = not is_numba_type_none(weight)
    have_sigma = not is_numba_type_none(sigma)
    
    # Make sure about the flags
    flags_match = matching_flag_flactory(have_flag_row)
    
    # Factory functions 
    uvw_factory = output_factory(have_uvw)
    time_centroid_factory = output_factory(have_time_centroid)
    exposure_factory = output_factory(have_exposure)
    weight_factory = output_factory(have_weight)
    sigma_factory = output_factory(have_sigma)
    
    # Factory
    time_centroid_adder = add_factory(have_time_centroid)
    exposure_adder = add_factory(have_exposure)
    
    uvw_adder = comp_add_factory(have_uvw)
    weight_adder = comp_add_factory(have_weight)
    sigma_adder = comp_add_factory(have_sigma)
    
    
    def impl(meta, ant1, ant2, flag_row=None,time_centroid=None, 
            exposure=None, uvw=None, weight=None, sigma=None):
        
        out_rows = meta.time.shape[0]
        print(out_rows)
        print(uvw.shape)
        counts = np.zeros(out_rows, dtype=np.uint32)
        ant1_avg = np.empty(out_rows, ant1.dtype)
        ant2_avg = np.empty(out_rows, ant2.dtype)
        
        uvw_avg = uvw_factory(out_rows, uvw)
        time_centroid_avg = time_centroid_factory(out_rows, time_centroid)
        exposure_avg = exposure_factory(out_rows, exposure)
        weight_avg = weight_factory(out_rows, weight)
        sigma_avg = sigma_factory(out_rows, sigma)
        
        
        
        return RowAverageOutput(ant1_avg, ant2_avg,
                                time_centroid_avg, exposure_avg, uvw_avg,
                                weight_avg, sigma_avg)
    return impl


@generated_jit(nopython=True, nogil=True, cache=True)
def baseline_row_chan_average():
    pass


AverageOutput = namedtuple("AverageOutput",
                           ["time", "interval", "flag_row"] +
                           _row_output_fields + _rowchan_output_fields)


@generated_jit(nopython=True, nogil=True, cache=True)
def baseline_time_and_channel(time, interval=None, antenna1, antenna2,
                     time_centroid=None, exposure=None, flag_row=None,
                     uvw=None, weight=None, sigma=None,
                     vis=None, flag=None,
                     weight_spectrum=None, sigma_spectrum=None,
                     bins_for_longest_baseline=1.0):
    
    # Verify data types
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
    
    def impl(time, interval=None, antenna1, antenna2,
                     time_centroid=None, exposure=None, flag_row=None,
                     uvw=None, weight=None, sigma=None,
                     vis=None, flag=None,
                     weight_spectrum=None, sigma_spectrum=None,
                     bins_for_longest_baseline=1.0):
    
        # Get the baseline row mapper data
        row_meta = baselibe_row_mapper(uvw, time, antenna1, antenna2,                                                          bins_for_longest_baseline=bins_for_longest_baseline)
    
    
        # Average the rows according to the meta data
        row_data = baseline_row_average(row_meta, antenna1, antenna2, interval, flag_row=flag_row,                                                      time_centroid=time_centroid, exposure=exposure, uvw=uvw, weight=weight, sigma=sigma)
        
        return AverageOutput(row_meta.time,row_data.interval, row_data.antenna1, row_data.antenna2,
                            row_data.time_centroid, row_data.exposure, row_data.uvw, row_data.weight, row_data.sigma)
    
    return impl
