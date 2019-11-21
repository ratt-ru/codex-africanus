# -*- coding: utf-8 -*-


import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest

from africanus.averaging.support import unique_time, unique_baselines
from africanus.averaging.time_and_channel_avg import time_and_channel
from africanus.averaging.bda_time_mapping import row_mapper

nchan = 16
ncorr = 4


@pytest.fixture
def time():
    return np.asarray([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])  # noqa


@pytest.fixture
def ant1():
    return np.asarray([0,   0,   1,   0,   0,   1,   2,   0,   0,   1],
                      dtype=np.int32)


@pytest.fixture
def ant2():
    return np.asarray([1,   2,   2,   0,   1,   2,   3,   0,   1,   2],
                      dtype=np.int32)


@pytest.fixture
def uvw():
    return np.asarray([[1.0,   1.0,  1.0],
                       [2.0,   2.0,  2.0],
                       [3.0,   3.0,  3.0],
                       [4.0,   4.0,  4.0],
                       [5.0,   5.0,  5.0],
                       [6.0,   6.0,  6.0],
                       [7.0,   7.0,  7.0],
                       [8.0,   8.0,  8.0],
                       [9.0,   9.0,  9.0],
                       [10.0, 10.0, 10.0]])


@pytest.fixture
def interval():
    data = np.asarray([1.9, 2.0, 2.1, 1.85, 1.95, 2.0, 2.05, 2.1, 2.05, 1.9])
    return 0.1 * data


@pytest.fixture
def weight(time):
    shape = (time.shape[0], ncorr)
    return np.arange(np.product(shape), dtype=np.float64).reshape(shape)


@pytest.fixture
def sigma(time):
    shape = (time.shape[0], ncorr)
    return np.arange(np.product(shape), dtype=np.float64).reshape(shape)


@pytest.fixture
def weight_spectrum(time):
    shape = (time.shape[0], nchan, ncorr)
    return np.arange(np.product(shape), dtype=np.float64).reshape(shape)


@pytest.fixture
def sigma_spectrum(time):
    shape = (time.shape[0], nchan, ncorr)
    return np.arange(np.product(shape), dtype=np.float64).reshape(shape)


@pytest.fixture
def frequency():
    return np.linspace(.856, 2*.856e9, nchan)


@pytest.fixture
def chan_width():
    return np.full(nchan, .856e9/nchan)


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


def _gen_testing_lookup(time, interval, ant1, ant2, flag_row, time_bin_secs,
                        row_meta):
    """
    Generates the same lookup as row_mapper, but different.

    Returns
    -------
    list of (float, (int, int), list of lists)
        Each tuple in the list corresponds to an output row, and
        is composed of `(avg_time, (ant1, ant2), effective_rows, nominal_rows)`

    """
    utime, _, time_inv, _ = unique_time(time)
    ubl, _, bl_inv, _ = unique_baselines(ant1, ant2)
    bl_time_lookup = np.full((ubl.shape[0], utime.shape[0]), -1,
                             dtype=np.int32)

    # Create the row index
    row_idx = np.arange(time.size)

    # Assign the row indices
    bl_time_lookup[bl_inv, time_inv] = row_idx

    # Create the time, baseline, row map
    time_bl_row_map = []

    # For each baseline, bin data that such that it fits within time_bin_secs
    # t1 - i1/2 + time_bin_secs < t2 - i2/2
    # where (t1, t2) and (i1, i2) are the times and intervals associated
    # with two different samples in the baseline.
    # Compute two different bins
    # 1. Effective row bin, which only includes unflagged rows
    #    unless the entire bin is flagged, in which case it includes flagged
    #    data
    # 2. Nominal row bin, which includes both flagged and unflagged rows

    for bl, (a1, a2) in enumerate(ubl):
        bl_row_idx = bl_time_lookup[bl, :]

        effective_bin_map = []
        effective_map = []

        nominal_bin_map = []
        nominal_map = []

        for ri in bl_row_idx:
            if ri == -1:
                continue

            half_int = 0.5 * interval[ri]

            # We're starting a new bin
            if len(nominal_map) == 0:
                bin_low = time[ri] - half_int
            # Reached passed the endpoint of the bin, start a new one
            elif time[ri] + half_int - bin_low > time_bin_secs:
                if len(effective_map) > 0:
                    effective_bin_map.append(effective_map)
                    nominal_bin_map.append(nominal_map)
                # No effective samples, the entire bin must be flagged
                elif len(nominal_map) > 0:
                    effective_bin_map.append(nominal_map)
                    nominal_bin_map.append(nominal_map)
                else:
                    raise ValueError("Zero-filled bin")

                effective_map = []
                nominal_map = []

            # Effective only includes unflagged samples
            if flag_row[ri] == 0:
                effective_map.append(ri)

            # Nominal includes all samples
            nominal_map.append(ri)

        # Add any remaining values
        if len(effective_map) > 0:
            effective_bin_map.append(effective_map)
            nominal_bin_map.append(nominal_map)
        # No effective samples, the entire bin must be flagged
        # so we add nominal samples
        elif len(nominal_map) > 0:
            effective_bin_map.append(nominal_map)
            nominal_bin_map.append(nominal_map)

        # Produce a (avg_time, bl, effective_rows, nominal_rows) tuple
        time_bl_row_map.extend((time[nrows].mean(), (a1, a2), erows, nrows)
                               for erows, nrows
                               in zip(effective_bin_map, nominal_bin_map))

    # Sort lookup sorted on averaged times
    return sorted(time_bl_row_map, key=lambda tup: tup[0])


def _calc_sigma(sigma, weight, idx):
    sigma = sigma[idx]
    weight = weight[idx]

    numerator = (sigma**2 * weight**2).sum(axis=0)
    denominator = weight.sum(axis=0)**2
    denominator[denominator == 0.0] = 1.0

    return np.sqrt(numerator / denominator)


@pytest.mark.parametrize("flagged_rows", [
    [], [8, 9], [4], [0, 1],
])
@pytest.mark.parametrize("time_bin_secs", [1, 2, 3, 4])
@pytest.mark.parametrize("chan_bin_size", [1, 3, 5])
def test_bda_averager(time, ant1, ant2, flagged_rows,
                      uvw, interval, weight, sigma,
                      frequency, chan_width,
                      vis, flag,
                      weight_spectrum, sigma_spectrum,
                      time_bin_secs, chan_bin_size):

    time_centroid = time
    exposure = interval

    vis = vis(time.shape[0], nchan, ncorr)
    flag = flag(time.shape[0], nchan, ncorr)

    flag_row = np.zeros(time.shape, dtype=np.uint8)

    # flagged_row and flag should agree
    flag_row[flagged_rows] = 1
    flag[flagged_rows, :, :] = 1

    row_meta = row_mapper(time, interval, ant1, ant2, uvw,
                          flag_row, time_bin_secs)


import numpy as np
import pytest


@pytest.fixture
def wsrt_ants():
    """ Westerbork antenna positions """
    return np.array([
           [3828763.10544699,   442449.10566454,  5064923.00777],
           [3828746.54957258,   442592.13950824,  5064923.00792],
           [3828729.99081359,   442735.17696417,  5064923.00829],
           [3828713.43109885,   442878.2118934,  5064923.00436],
           [3828696.86994428,   443021.24917264,  5064923.00397],
           [3828680.31391933,   443164.28596862,  5064923.00035],
           [3828663.75159173,   443307.32138056,  5064923.00204],
           [3828647.19342757,   443450.35604638,  5064923.0023],
           [3828630.63486201,   443593.39226634,  5064922.99755],
           [3828614.07606798,   443736.42941621,  5064923.],
           [3828609.94224429,   443772.19450029,  5064922.99868],
           [3828601.66208572,   443843.71178407,  5064922.99963],
           [3828460.92418735,   445059.52053929,  5064922.99071],
           [3828452.64716351,   445131.03744105,  5064922.98793]],
        dtype=np.float64)


def test_uvw_convert(wsrt_ants):
    from africanus.rime.tests.test_parangles import _observation_endpoints
    from astropy.coordinates import (EarthLocation, SkyCoord,
                                     AltAz, CIRS)
    from astropy.time import Time
    from astropy import units
    from africanus.util.astropy import UVW

    start, end = _observation_endpoints(2019, 10, 9, 8)

    # Convert from MJD second to MJD
    times = np.linspace(start, end, 32)
    times = Time(times / 86400.00, format='mjd', scale='utc')

    mapos = wsrt_ants.mean(axis=0)[:, None]
    mapos = EarthLocation.from_geocentric(*mapos, unit='m')

    # altaz = AltAz(location=ap[None, :], obstime=times[:, None])
    phase_centre = SkyCoord(ra=60, dec=0, unit=units.deg, frame='fk5')
    uvw = UVW(location=mapos[None, :], obstime=times[:, None],
              phase=phase_centre)
    print(uvw)



def __test_uvw_convert_2(wsrt_ants):
    x, y, z = np.hsplit(wsrt_ants, 3)

    from astropy.time import Time
    from astropy import units
    from astropy.coordinates import Angle
    from africanus.rime.tests.test_parangles import _observation_endpoints

    start, end = _observation_endpoints(2019, 10, 9, 8)

    # Convert from MJD second to MJD
    times = np.linspace(start, end, 32)
    times = Time(times / 86400.00, format='mjd', scale='utc')



    ha = Angle("3h0m0s")
    dec = 10*units.deg

    a1, a2 = np.triu_indices(wsrt_ants.shape[0], 1)

    # Two rotations:
    #  1. by 'ha' along the z axis
    #  2. by '90-dec' along the u axis
    u = x * np.cos(ha) - y * np.sin(ha)
    v0 = x * np.sin(ha) + y * np.cos(ha)
    w = z * np.sin(dec) - v0 * np.cos(dec)
    v = z * np.cos(dec) + v0 * np.sin(dec)

    uvw = np.hstack([u, v, w])

    print(uvw[a1] - uvw[a2])

