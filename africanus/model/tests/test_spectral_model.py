# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import pytest


def test_spectral_model():
    from africanus.model import spectral_model

    nsrc = 10
    nchan = 16

    I = np.random.random(nsrc)
    frequency = np.linspace(.856e9, 2*.856e9, nchan)
    spi = 0.7 + np.random.random(nsrc) * 0.2
    ref_freq = np.full(I.shape, 3*.856e9/2, dtype=frequency.dtype)

    spectral_model(I, spi, ref_freq, frequency)
