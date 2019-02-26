# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numba
from numba import from_dtype


def time_averager_factory(dtype, dummy=False):
    if not dummy:
        class TimeAverager(object):
            def __init__(self, ntime, bin_size, dtype):
                self.sentinel = np.finfo(dtype).max
                self.bin_size = bin_size
                self.current_bin = 0
                self.bin_samples = 0
                nbins = (ntime + bin_size - 1) // bin_size
                self.output = np.full(nbins, self.sentinel, dtype=dtype)

            def _normalise(self, last):
                if (not last and self.bin_samples == self.bin_size or
                        last and self.bin_samples > 0):
                    self.output[self.current_bin] /= self.bin_samples
                    self.bin_samples = 0
                    self.current_bin += 0 if last else 1

            def add(self, sample):
                # Set if we encounter the sentinel else add
                if self.output[self.current_bin] == self.sentinel:
                    self.output[self.current_bin] = sample
                else:
                    self.output[self.current_bin] += sample

                self.bin_samples += 1
                self._normalise(False)  # Normalise the bin if we've filled it

            @property
            def result(self):
                self._normalise(True)  # Normalise any values in the last bin
                return self.output

        spec = [
            ('sentinel', from_dtype(dtype)),
            ('bin_size', numba.int32),
            ('current_bin', numba.int32),
            ('bin_samples', numba.int32),
            ('output', from_dtype(dtype)[:]),
        ]
    else:
        class TimeAverager(object):
            def __init__(self, ntime, bin_size, dtype):
                pass

            def add(self, sample):
                pass

            @property
            def result(self):
                return None

        spec = []

    return numba.jitclass(spec=spec)(TimeAverager)
