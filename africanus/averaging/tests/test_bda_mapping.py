# -*- coding: utf-8 -*-

import numpy as np
import pytest

from africanus.constants import c as lightspeed

import numpy as np
import pytest

from astropy.coordinates import (EarthLocation,
                                    SkyCoord,
                                    AltAz, CIRS)
from astropy.time import Time
from astropy import units

from pyrap.measures import measures
from pyrap.quanta import quantity as q

def synthesize_uvw(antenna_positions, time, phase_dir,
                   auto_correlations=True):
    """
    Synthesizes new UVW coordinates based on time according to
    NRAO CASA convention (same as in fixvis)
    User should check these UVW coordinates carefully:
    if time centroid was used to compute
    original uvw coordinates the centroids
    of these new coordinates may be wrong, depending on whether
    data timesteps were heavily flagged.
    """

    dm = measures()
    epoch = dm.epoch("UT1", q(time[0], "s"))
    ref_dir = dm.direction("j2000",
                           q(phase_dir[0], "rad"),
                           q(phase_dir[1], "rad"))
    ox, oy, oz = antenna_positions[0]
    obs = dm.position("ITRF", q(ox, "m"), q(oy, "m"), q(oz, "m"))

    # Setup local horizon coordinate frame with antenna 0 as reference position
    dm.do_frame(obs)
    dm.do_frame(ref_dir)
    dm.do_frame(epoch)

    ant1, ant2 = np.triu_indices(antenna_positions.shape[0],
                                 0 if auto_correlations else 1)

    ant1 = ant1.astype(np.int32)
    ant2 = ant2.astype(np.int32)

    ntime = time.shape[0]
    nbl = ant1.shape[0]
    rows = ntime * nbl
    uvw = np.empty((rows, 3), dtype=np.float64)

    # For each timestep
    for ti, t in enumerate(time):
        epoch = dm.epoch("UT1", q(t, "s"))
        dm.do_frame(epoch)

        ant_uvw = np.zeros_like(antenna_positions)

        # Calculate antenna UVW positions
        for ai, (x, y, z) in enumerate(antenna_positions):
            bl = dm.baseline("ITRF",
                             q([x, ox], "m"),
                             q([y, oy], "m"),
                             q([z, oz], "m"))

            ant_uvw[ai] = dm.to_uvw(bl)["xyz"].get_value()[0:3]

        # Now calculate baseline UVW positions
        # noting that ant1 - ant2 is the CASA convention
        base = ti*nbl
        uvw[base:base + nbl, :] = ant_uvw[ant1] - ant_uvw[ant2]

    return ant1, ant2, uvw


@pytest.fixture
def time():
    return np.array([
        5.03373334e+09,   5.03373334e+09,   5.03373335e+09,
        5.03373336e+09,   5.03373337e+09,   5.03373338e+09,
        5.03373338e+09,   5.03373339e+09,   5.03373340e+09,
        5.03373341e+09,   5.03373342e+09,   5.03373342e+09,
        5.03373343e+09,   5.03373344e+09,   5.03373345e+09,
        5.03373346e+09,   5.03373346e+09,   5.03373347e+09,
        5.03373348e+09,   5.03373349e+09,   5.03373350e+09,
        5.03373350e+09,   5.03373351e+09,   5.03373352e+09,
        5.03373353e+09,   5.03373354e+09,   5.03373354e+09,
        5.03373355e+09,   5.03373356e+09,   5.03373357e+09,
        5.03373358e+09,   5.03373358e+09,   5.03373359e+09,
        5.03373360e+09,   5.03373361e+09,   5.03373362e+09])


@pytest.fixture
def interval():
    return np.array([
        7.99661697,  7.99661697,  7.99661697,  7.99661697,  7.99661697,
        7.99661697,  7.99661697,  7.99661697,  7.99661697,  7.99661697,
        7.99661697,  7.99661697,  7.99661697,  7.99661697,  7.99661697,
        7.99661697,  7.99661697,  7.99661697,  7.99661697,  7.99661697,
        7.99661697,  7.99661697,  7.99661697,  7.99661697,  7.99661697,
        7.99661697,  7.99661697,  7.99661697,  7.99661697,  7.99661697,
        7.99661697,  7.99661697,  7.99661697,  7.99661697,  7.99661697,
        7.99661697])


@pytest.fixture
def ants():
    return np.array([
       [5109224.29038545,  2006790.35753831, -3239100.60907827],
       [5109247.7157809,   2006736.96831224, -3239096.13639116],
       [5109222.76106102,  2006688.94849795, -3239165.94167899],
       [5109101.13948279,  2006650.38001812, -3239383.31891167],
       [5109132.81491624,  2006798.06346825, -3239242.1849703],
       [5109046.33257705,  2006823.98423929, -3239363.78875328],
       [5109095.03238529,  2006898.89823927, -3239239.95261248],
       [5109082.8918671,  2007045.24176653, -3239169.09131402],
       [5109139.53289849,  2006992.25575245, -3239111.37956843],
       [5109368.62360157,  2006509.64851116, -3239043.72292735],
       [5109490.75061883,  2006708.38364351, -3238726.60664016],
       [5109310.2977957,  2007017.0371345, -3238823.74833534],
       [5109273.32322089,  2007083.40054198, -3238841.20407917],
       [5109233.60247272,  2007298.47483172, -3238770.86653967],
       [5109514.1862076,  2007536.98018719, -3238177.03761655],
       [5109175.83425585,  2007164.6225741, -3238946.9157957],
       [5109093.99046283,  2007162.9306937, -3239078.77530747],
       [5108965.29396408,  2007106.07798817, -3239319.10626408],
       [5108993.64502175,  2006679.78785901, -3239536.3704696],
       [5109111.46526165,  2006445.98820889, -3239491.95574845],
       [5109486.39986795,  2006225.48918911, -3239031.01140517],
       [5109925.48993011,  2006111.83927162, -3238401.39137192],
       [5110109.89167353,  2005177.90721032, -3238688.71487862],
       [5110676.49309192,  2005793.15912039, -3237408.15958056],
       [5109284.52911273,  2006201.59095546, -3239366.63085706],
       [5111608.06713389,  2004721.2262196, -3236602.97648213],
       [5110840.88031587,  2003560.05835788, -3238544.12229424],
       [5109666.45350777,  2004767.93425934, -3239646.10724868],
       [5108767.23563213,  2007556.54497446, -3239354.53798391],
       [5108927.44284297,  2007973.80069955, -3238840.15661171],
       [5110746.29394702,  2007713.62376395, -3236109.83563026],
       [5109561.42891041,  2009946.10154943, -3236606.07622565],
       [5108335.37384839,  2010410.68719286, -3238271.56790951],
       [5107206.7556267,  2009680.79691055, -3240512.45932645],
       [5108231.34344288,  2006391.59690538, -3240926.75417832],
       [5108666.77102205,  2005032.4814725, -3241081.69797118]])


@pytest.fixture
def phase_dir():
    return np.array([5.1461782, -1.11199629])


@pytest.fixture
def ref_freq():
    return 1.284e9

@pytest.fixture
def chan_width():
    return np.full(4096, 208984.375)

@pytest.fixture
def chan_freq(chan_width):
    return .856e9 + np.cumsum(np.concatenate([[0], chan_width[1:]]))

@pytest.fixture
def ref_freq():
    return 1.284e9


def test_atemkeng_bda_mapper(time, ants, interval, phase_dir,
                             ref_freq, chan_freq, chan_width):
    time = np.unique(time)
    from africanus.averaging.bda_mapping import atemkeng_mapper, partition_frequency
    ant1, ant2, uvw = synthesize_uvw(ants, time, phase_dir, False)

    decorrelation = 0.95
    max_uvw_dist = np.sqrt(np.sum(uvw**2, axis=1)).max()

    # spw_chan_freqs = partition_frequency(16, chan_freq, chan_width,
    #                                      ref_freq, max_uvw_dist,
    #                                      decorrelation=decorrelation)

    # print(spw_chan_freqs)

    nbl = ant1.shape[0]
    ntime = time.shape[0]

    time = np.repeat(time, nbl)
    interval = np.repeat(interval, nbl)
    ant1 = np.tile(ant1, ntime)
    ant2 = np.tile(ant2, ntime)

    max_uvw_dist = np.sqrt(np.sum(uvw**2, axis=1)).max()

    atemkeng_mapper(time, interval, ant1, ant2, uvw,
                    ref_freq, chan_freq, chan_width,
                    max_uvw_dist,
                    lm_max=1.0, decorrelation=decorrelation)

    print(max_uvw_dist)
