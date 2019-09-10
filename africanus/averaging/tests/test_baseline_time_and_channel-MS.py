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
from baseline_time_and_channel_avg import baseline_time_and_channel

import argparse
from pyrap.tables import table

#import .ms_manipulator


parser = argparse.ArgumentParser()
parser.add_argument("in_ms", help="Input MS")
parser.add_argument("out_ms", help="Output MS")
args = parser.parse_args()

#MS you want to average
#inputMS = "MS-Multple-SPECTRAL-Windows/MeerKATDATA/1491291289.1GC.ms/"
inputMS = args.in_ms
#BDA MS, provide location and name, this will be created automatically when you run the script
outputMS = args.out_ms

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
sigma = tab.getcol("SIGMA")
# sigma_spectrum = tab.getcol("SIGMA_SPECTRUM")
vis = tab.getcol("DATA")
weight = tab.getcol("WEIGHT")
# weight_spectrum = tab.getcol("WEIGHT_SPECTRUM")
scan = tab.getcol("SCAN_NUMBER")

tab.close()

print("Original time \n", time.shape)
print("Orginal interval \n", interval.shape)
print("Orginal antenna1 \n", ant1.shape)
print("Orginal antenna2", ant2.shape)
print("Orginal time_centroid \n", time_centroid.shape)
print("Orginal exposure \n", exposure.shape)
print("Orginal flag_row \n", flag_row.shape)
print("Orginal uvw \n", uvw.shape)
print("Orginal weight \n", weight.shape)
print("Orginal sigma \n", sigma.shape)
print("Orginal vis \n", vis.shape)
print("Orginal vis dtype \n", vis.dtype)
print("Orginal flag \n", flag.shape)

print("Running test_baseline_time_and_channel")
avg = baseline_time_and_channel(time, interval, ant1, ant2, time_centroid=time_centroid, exposure=exposure,
                                flag_row=flag_row, uvw=uvw, weight=weight, sigma=sigma, vis=vis, flag=flag, 
                                weight_spectrum=None,sigma_spectrum=None, bins_for_longest_baseline=1.0, 
                                baseline_chan_bin_size=1)

# print("avg.time \n", avg.time.shape)
# print("avg.flag_row \n", avg.flag_row.shape)
# print("avg.interval \n", avg.interval.shape)
# print("avg.antenna1 \n", avg.antenna1.shape)
# print("avg.antenna2", avg.antenna2.shape)
# print("avg.time_centroid \n", avg.time_centroid.shape)
# print("avg.exposure \n", avg.exposure.shape)
# print("avg.uvw \n", avg.uvw.shape)
# print("avg.weight \n", avg.weight.shape)
# print("avg.sigma \n", avg.sigma.shape)
# print("avg.vis \n", avg.vis.shape)
# print("avg.flag \n", avg.flag.shape)
# print("avg.weight_spectrum \n", avg.weight_spectrum.shape)
# print("avg.sigma_spectrum \n", avg.sigma_spectrum.shape)


# na = np.max(A0) + 1
# nbl = na * (na - 1) / 2
# table_desc, dm_info = ms_manipulator.kat_ms_desc_and_dminfo(nbl=nbl, model_data=True)

# ms_manipulator.create_ms(outputMS, table_desc, dminfo)

# ms_dict = ms_manipulator.populate_main_dict_time(inputMS, avg.scan, avg.desc, avg.uvw, avg.flag_row, avg.flag, avg.antenna1, avg.antenna2, avg.interval, avg.exposure, avg.time, avg.time_centroid, avg.vis, avg.weight)

# ms_manipulator.write_dict(ms_dict, outputMS)

