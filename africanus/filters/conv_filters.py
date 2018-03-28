# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numba
import numpy as np

ConvolutionFilter = collections.namedtuple("ConvolutionFilter",
        ['half_sup', 'oversample',
        'full_sup_wo_padding', 'full_sup',
        'no_taps', 'filter_taps'])

def convolution_filter(half_support, oversampling_factor, filter_type):
    """
    Create a 1D Convolution Filter suitable
    for use with gridding and degridding functions.

    Parameters
    ----------
    half_support : integer
        Half support (N) of the filter. The filter has a
        full support of N*2 + 1 taps
    oversampling_factor : integer
        Number of spaces in-between grid-steps
        (improves gridding/degridding accuracy)
    filter_type : {'sinc', 'box', 'gaussian'}
        Filter type

    Returns
    -------
    :class:`ConvolutionFilter`
        namedtuple containing filter attributes
    """
    full_sup_wo_padding = (half_support * 2 + 1)
    full_sup = full_sup_wo_padding + 2 #+ padding
    no_taps = full_sup + (full_sup - 1) * (oversampling_factor - 1)

    taps = np.arange(no_taps)/float(oversampling_factor) - full_sup / 2

    if filter_type == 'box':
        filter_taps = np.empty_like(taps, dtype=np.float64)
        condition = (taps >= -0.5) & (taps <= 0.5)
        filter_taps[condition] = 1
        filter_taps[np.invert(condition)] = 0
    elif filter_type == 'sinc':
        filter_taps = np.sinc(taps)
    elif filter_type == 'gaussian_sinc':
        alpha_1 = 1.55
        alpha_2 = 2.52
        taps_eps = taps + 1e-11

        filter_taps = np.exp(-(taps/alpha_2)**2)
        filter_taps *= np.sin(np.pi*taps_eps/alpha_1)
        filter_taps /= np.pi*taps_eps
    else:
        raise ValueError("Expected one of 'box','sinc' or 'gaussian_sinc'")

    return ConvolutionFilter(half_support, oversampling_factor,
        full_sup_wo_padding, full_sup, no_taps, filter_taps)
