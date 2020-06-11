# -*- coding: utf-8 -*-

import numpy as np
import pytest

from africanus.averaging.tests.test_bda_mapping import (  # noqa: F401
                            synthesize_uvw,
                            time,
                            interval,
                            ants,
                            phase_dir,
                            ref_freq,
                            chan_width,
                            chan_freq)

from africanus.averaging.bda_mapping import atemkeng_mapper
from africanus.averaging.bda_avg import row_average, row_chan_average


@pytest.fixture
def vis():
    def _vis(row, chan, fcorrs):
        flat_vis = (np.arange(row*chan*fcorrs, dtype=np.float64) +
                    np.arange(1, row*chan*fcorrs+1, dtype=np.float64)*1j)

        return flat_vis.reshape(row, chan, fcorrs)

    return _vis


@pytest.fixture
def flag():
    def _flag(row, chan, fcorrs):
        return np.random.randint(0, 2, (row, chan, fcorrs))

    return _flag


def test_bda_avg(time, interval, ants,   # noqa: F811
                 phase_dir, ref_freq,    # noqa: F811
                 chan_freq, chan_width,  # noqa: F811
                 vis, flag):             # noqa: F811
    time = np.unique(time)
    ant1, ant2, uvw = synthesize_uvw(ants[:14], time, phase_dir, False)

    nchan = chan_width.shape[0]
    ncorr = 4

    nbl = ant1.shape[0]
    ntime = time.shape[0]

    time = np.repeat(time, nbl)
    interval = np.repeat(interval, nbl)
    ant1 = np.tile(ant1, ntime)
    ant2 = np.tile(ant2, ntime)
    flag_row = np.zeros(time.shape[0], dtype=np.int8)

    decorrelation = 0.95
    max_uvw_dist = np.sqrt(np.sum(uvw**2, axis=1)).max()

    meta = atemkeng_mapper(time, interval, ant1, ant2, uvw,
                           ref_freq, max_uvw_dist, chan_width,
                           flag_row=flag_row, lm_max=1.0,
                           decorrelation=decorrelation)

    time_centroid = time
    exposure = interval

    row_avg = row_average(meta, ant1, ant2, flag_row,  # noqa: F841
                          time_centroid, exposure,
                          uvw, weight=None, sigma=None)

    vis = vis(time.shape[0], nchan, ncorr)
    flag = flag(time.shape[0], nchan, ncorr)
    weight_spectrum = np.random.random(size=flag.shape).astype(np.float64)
    sigma_spectrum = np.random.random(size=flag.shape).astype(np.float64)

    row_chan = row_chan_average(meta,  # noqa: F841
                                flag_row=flag_row,
                                vis=vis, flag=flag,
                                weight_spectrum=weight_spectrum,
                                sigma_spectrum=sigma_spectrum)
