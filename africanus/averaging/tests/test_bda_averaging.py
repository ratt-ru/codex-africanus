# -*- coding: utf-8 -*-

import numpy as np

from africanus.averaging.tests.test_bda_mapping import (synthesize_uvw,
                                                        time,
                                                        interval,
                                                        ants,
                                                        phase_dir,
                                                        ref_freq,
                                                        chan_width,
                                                        chan_freq)

from africanus.averaging.bda_mapping import atemkeng_mapper
from africanus.averaging.bda_avg import row_average


def test_bda_avg(time, interval, ants, phase_dir, ref_freq, chan_freq, chan_width):
    time = np.unique(time)
    ant1, ant2, uvw = synthesize_uvw(ants, time, phase_dir, False)

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
                           ref_freq, max_uvw_dist, flag_row,
                           lm_max=1.0, decorrelation=decorrelation)

    time_centroid = time
    exposure = interval

    avg = row_average(meta, time, interval, flag_row,
                      time_centroid, exposure, uvw,
                      weight=None, sigma=None)

    print(avg.uvw)
