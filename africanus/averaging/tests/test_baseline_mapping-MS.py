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
from baseline_time_and_channel_mapping import baseline_row_mapper

import argparse
from pyrap.tables import table

parser = argparse.ArgumentParser()
parser.add_argument("in_ms", help="Input MS")
# parser.add_argument("out_ms", help="Output MS")
args = parser.parse_args()

#MS you want to average
#inputMS = "MS-Multple-SPECTRAL-Windows/MeerKATDATA/1491291289.1GC.ms/"
inputMS = args.in_ms
#BDA MS, provide location and name, this will be created automatically when you run the script
# outputMS = args.out_ms

tab = table(inputMS, ack=True, readonly=False)
desc = tab.getcol("DATA_DESC_ID")
uvw = tab.getcol("UVW")
flag_row = tab.getcol("FLAG_ROW")
flag = tab.getcol("FLAG")
ant1 = tab.getcol("ANTENNA1")
ant2 = tab.getcol("ANTENNA2")
interval = tab.getcol("INTERVAL")
exposure = tab.getcol("EXPOSURE")
time = tab.getcol("TIME")
time_centroid = tab.getcol("TIME_CENTROID")
vis_data = tab.getcol("DATA")
weight = tab.getcol("WEIGHT")
scan = tab.getcol("SCAN_NUMBER")


def flag_row_factory(nrows, flagged_rows):
    flag_row = np.zeros(nrows, dtype=np.uint8)

    if flagged_rows is not None:
        flag_row[flagged_rows] = 1

    return flag_row


bins_for_longest_baseline = 1 # , 0.2, 1, 2, 4])
flagged_rows = None #, [0, 1], [2, 4], range(10)])

def test_baseline_row_mapper(uvw, time, ant1, ant2, flag_row, bins_for_longest_baseline):
    
#     utime, _, time_inv, _ = unique_time(time)
#     ubl, _, bl_inv, _ = unique_baselines(ant1, ant2)
#     mask = np.full((ubl.shape[0], utime.shape[0]), -1, dtype=np.int32)

#     mask[bl_inv, time_inv] = np.arange(time.size)

#     flag_row = flag_row_factory(time.size, flagged_rows)

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
test_baseline_row_mapper(uvw, time, ant1, ant2, flag_row, bins_for_longest_baseline)

tab.close()