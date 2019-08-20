# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest

import sys
sys.path.insert(0, '/Users/smasoka/Varsity/codex-africanus/africanus/averaging/')
from support import unique_time, unique_baselines
from baseline_time_and_channel_mapping import (baseline_row_mapper,baseline_chan_mapper)

time = np.asarray([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
ant1 = np.asarray([0,   0,   1,   0,   0,   1,   2,   0,   0,   1], dtype=np.int32)
ant2 = np.asarray([1,   2,   2,   0,   1,   2,   3,   0,   1,   2], dtype=np.int32)
uvw = np.asarray([[1.0,   1.0,  1.0],
                  [2.0,   2.0,  2.0],
                  [3.0,   3.0,  3.0],
                  [4.0,   4.0,  4.0],
                  [1.0,   1.0,  1.0],
                  [3.0,   3.0,  3.0],
                  [5.0,   5.0,  5.0],
                  [4.0,   4.0,  4.0],
                  [1.0,   1.0,  1.0],
                  [3.0,   3.0,  3.0]])

interval = np.asarray([1.9, 2.0, 2.1, 1.85, 1.95, 2.0, 2.05, 2.1, 2.05, 1.9]) * 0.1


def flag_row_factory(nrows, flagged_rows):
    flag_row = np.zeros(nrows, dtype=np.uint8)

    if flagged_rows is not None:
        flag_row[flagged_rows] = 1

    return flag_row


bins_for_longest_baseline = 1 # , 0.2, 1, 2, 4])
flagged_rows = None #, [0, 1], [2, 4], range(10)])

def test_baseline_row_mapper(uvw, time, ant1, ant2, flagged_rows, bins_for_longest_baseline):
    
    utime, _, time_inv, _ = unique_time(time)
    ubl, _, bl_inv, _ = unique_baselines(ant1, ant2)
    mask = np.full((ubl.shape[0], utime.shape[0]), -1, dtype=np.int32)

    mask[bl_inv, time_inv] = np.arange(time.size)

    flag_row = flag_row_factory(time.size, flagged_rows)

    ret = baseline_row_mapper(uvw, time, ant1, ant2, flag_row, bins_for_longest_baseline)

    # For TIME AND INTERVAL, flagged inputs can
    # contribute to unflagged outputs
    new_time = np.zeros_like(ret.time)
    counts = np.zeros(ret.time.shape, dtype=np.uint32)
    np.add.at(new_time, ret.map, time)
    np.add.at(counts, ret.map, 1)

    print("ret.time\n", ret.time)
    print("new_time\n", new_time)
    print("counts", counts)
    assert_array_equal(ret.time, new_time / counts)

    # Now recalculate time_avg using the row_map
    new_tc = np.zeros_like(ret.time)
    counts = np.zeros(ret.time.shape, dtype=np.uint32)

    sel = flag_row == ret.flag_row[ret.map]
    np.add.at(new_tc, ret.map[sel], time[sel])
    np.add.at(counts, ret.map[sel], 1)

    ant1_avg = np.empty(ret.time.shape, dtype=ant1.dtype)
    ant2_avg = np.empty(ret.time.shape, dtype=ant2.dtype)
    ant1_avg[ret.map[sel]] = ant1[sel]
    ant2_avg[ret.map[sel]] = ant2[sel]

    # Do it a different way
    new_tc2 = np.zeros_like(ret.time)
    counts2 = np.zeros(ret.time.shape, dtype=np.uint32)

    for ri, ro in enumerate(ret.map):
        if flag_row[ri] == 1 and ret.flag_row[ro] == 1:
            new_tc2[ro] += time[ri]
            counts2[ro] += 1
        elif flag_row[ri] == 0 and ret.flag_row[ro] == 0:
            new_tc2[ro] += time[ri]
            counts2[ro] += 1


    assert_array_almost_equal(new_tc / counts, new_tc2 / counts2)

# Call test_baseline_row_mapper
#test_baseline_row_mapper(uvw, time, ant1, ant2, flagged_rows, bins_for_longest_baseline)


baseline_chan_bin_size = 1
nchan = 64
def test_baseline_chan_mapper(uvw, ant1, ant2, nchan, baseline_chan_bin_size):
    bl_chan_map, bl_chan_count = baseline_chan_mapper(uvw, ant1, ant2, nchan, baseline_chan_bin_size)
    
    print(bl_chan_map.shape)
    print(bl_chan_count)

test_baseline_chan_mapper(uvw, ant1, ant2, nchan, baseline_chan_bin_size)

