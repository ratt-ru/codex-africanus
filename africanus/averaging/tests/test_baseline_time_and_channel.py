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


nchan = 16
ncorr = 4

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
shape = (time.shape[0], ncorr)
weight = np.arange(np.product(shape), dtype=np.float64).reshape(shape)
sigma = np.arange(np.product(shape), dtype=np.float64).reshape(shape)
shape = (time.shape[0], nchan, ncorr)
weight_spectrum = np.arange(np.product(shape), dtype=np.float64).reshape(shape)
sigma_spectrum = np.arange(np.product(shape), dtype=np.float64).reshape(shape)
row = time.shape[0]
vis = (np.arange(row*nchan*ncorr, dtype=np.float32) +
      np.arange(1, row*nchan*ncorr+1, dtype=np.float32)*1j)
vis = vis.reshape(row, nchan, ncorr)
flag = np.random.randint(0, 2, (row, nchan, ncorr))
flag_row = np.zeros(time.shape, dtype=np.uint8)
flagged_rows = None #, [0, 1], [2, 4], range(10)])

flag_row[flagged_rows] = 1
flag_row = np.all(flag, axis=(1, 2))



print("Original time \n", time)
print("Orginal interval \n", interval)
print("Orginal antenna1 \n", ant1)
print("Orginal antenna2", ant2)
print("Orginal uvw \n", uvw)
print("Orginal weight \n", weight)
print("Orginal sigma \n", sigma)

print("Running test_baseline_time_and_channel")
avg = baseline_time_and_channel(time, interval, ant1, ant2,
                     time_centroid=time, exposure=interval, flag_row=flag_row,
                     uvw=uvw, weight=weight, sigma=sigma,vis=vis, flag=flag, weight_spectrum=weight_spectrum, sigma_spectrum=sigma_spectrum,
                     bins_for_longest_baseline=1.0)

print("avg.time \n", avg.time)
print("avg.interval \n", avg.interval)
print("avg.antenna1 \n", avg.antenna1)
print("avg.antenna2", avg.antenna2)
print("avg.uvw \n", avg.uvw)
print("avg.weight \n", avg.weight)
print("avg.sigma \n", avg.sigma)


