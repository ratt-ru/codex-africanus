# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numba
from numba.numpy_support import from_dtype


def time_averager_spec(ntime, bin_size, dtype):
    return [
        ('bin_size', numba.int32),
        ('current_bin', numba.int32),
        ('bin_samples', numba.int32),
        ('output', from_dtype(dtype)[:]),
    ]


class TimeAverager(object):
    def __init__(self, ntime, bin_size, dtype):
        self.bin_size = bin_size
        self.current_bin = 0
        self.bin_samples = 0
        nbins = (ntime + bin_size - 1) // bin_size
        self.output = np.zeros(nbins, dtype=dtype)

    def normalise(self, last):
        if (not last and self.bin_samples == self.bin_size or
                last and self.bin_samples > 0):
            self.output[self.current_bin] /= self.bin_samples
            self.bin_samples = 0
            self.current_bin += 0 if last else 1

    def add(self, sample):
        self.output[self.current_bin] += sample  # Add the sample
        self.bin_samples += 1
        self.normalise(False)  # Normalise the bin if we've filled it

    @property
    def result(self):
        self.normalise(True)  # Normalise any values in the last bin
        return self.output


def vis_averager_spec(rows, chans, corrs, row_bin_size, chan_bin_size, dtype):
    return [
        ('row_bin_size', numba.int32),
        ('chan_bin_size', numba.int32),
        ('row_bin', numba.int32),
        ('chan_bin', numba.int32),
        ('row_samples', numba.int32),
        ('chan_samples', numba.int32),
        ('output', from_dtype(dtype)[:, :, :]),
        ('scratch',  from_dtype(dtype)[:, :])
    ]


class VisAverager(object):
    def __init__(self, rows, chans, corrs, row_bin_size, chan_bin_size, dtype):
        self.row_bin_size = row_bin_size
        self.chan_bin_size = chan_bin_size
        self.row_bin = 0
        self.chan_bin = 0
        self.row_samples = 0
        self.chan_samples = 0

        out_shape = ((rows + row_bin_size - 1) // row_bin_size,
                     (chans + chan_bin_size - 1) // chan_bin_size,
                     corrs)

        self.output = np.zeros(out_shape, dtype=dtype)
        self.scratch = np.zeros(out_shape[1:], dtype=dtype)

    def normalise(self, last):
        pass

    def add(self, chan_corrs):
        pass


_averager_class_lookup = {
    'TimeAverager': (TimeAverager, time_averager_spec),
    'VisAverager': (VisAverager, vis_averager_spec),
}


def averaging_factory(averager, *args, **kwargs):
    try:
        py_cls, spec_fn = _averager_class_lookup[averager]
    except KeyError:
        raise ValueError("'%s' averager not registered. %s available." %
                         (averager, _averager_class_lookup.keys()))

    # Get the class spec and create the numba class
    spec = spec_fn(*args, **kwargs)
    numba_cls = numba.jitclass(spec=spec)(py_cls)

    # Instantiate the averager
    return numba_cls(*args, **kwargs)
