# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from africanus.averaging.support import unique_time, unique_baselines
from africanus.averaging.time_and_channel_avg import time_and_channel
from africanus.averaging.time_and_channel_mapping import (row_mapper,
                                                          channel_mapper)

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
def vis():
    def _vis(row, chan, fcorrs):
        flat_vis = (np.arange(row*chan*fcorrs, dtype=np.float32) +
                    np.arange(1, row*chan*fcorrs+1, dtype=np.float32)*1j)

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
        is composed of `(avg_time, (ant1, ant2), binned_input_rows)`

    """
    utime, _, time_inv, _ = unique_time(time)
    ubl, _, bl_inv, _ = unique_baselines(ant1, ant2)
    bl_time_lookup = np.full((ubl.shape[0], utime.shape[0]), -1,
                             dtype=np.int32)

    # See :func:`row_mapper` docs for a discussion of this
    # but we use it to produce different bins for the
    # TIME_CENTROID/EXPOSURE and TIME/INTERVAL cases
    sel = flag_row == row_meta.flag_row[row_meta.map]
    invalid_rows = set(np.where(~sel)[0])

    # Create the row index
    row_idx = np.arange(time.size)

    # Assign the row indices
    bl_time_lookup[bl_inv, time_inv] = row_idx

    # Create the time, baseline, row map
    time_bl_row_map = []

    for bl, (a1, a2) in enumerate(ubl):
        bl_row_idx = bl_time_lookup[bl, :]

        bin_map = []
        current_map = []

        flagged_bin_map = []
        current_flagged_map = []

        for ri in bl_row_idx:
            if ri == -1:
                continue

            half_exp = 0.5 * interval[ri]

            # We're starting a new bin
            if len(current_map) == 0:
                bin_low = time[ri] - half_exp
            # Reached passed the endpoint of the bin, start a new one
            elif time[ri] + half_exp - bin_low > time_bin_secs:
                bin_map.append(current_map)
                flagged_bin_map.append(current_flagged_map)
                current_map = []
                current_flagged_map = []

            # add current row to the bin
            if ri not in invalid_rows:
                current_map.append(ri)

            current_flagged_map.append(ri)

        # Add any remaining maps
        if len(current_map) > 0:
            bin_map.append(current_map)
            flagged_bin_map.append(current_flagged_map)

        # Produce a (avg_time, bl, rows, rows_including_flag_rows) tuple
        time_bl_row_map.extend((time[rows].mean(), (a1, a2), rows, frows)
                               for rows, frows
                               in zip(bin_map, flagged_bin_map))

    # Sort lookup sorted on averaged times
    return sorted(time_bl_row_map, key=lambda tup: tup[0])


@pytest.mark.parametrize("flagged_rows", [
    [], [8, 9], [4], [0, 1],
])
@pytest.mark.parametrize("time_bin_secs", [1, 2, 3, 4])
@pytest.mark.parametrize("chan_bin_size", [1, 3, 5])
def test_averager(time, ant1, ant2, flagged_rows,
                  uvw, interval, weight, sigma,
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

    row_meta = row_mapper(time, interval, ant1, ant2, flag_row, time_bin_secs)
    chan_map, chan_bins = channel_mapper(nchan, chan_bin_size)

    time_bl_row_map = _gen_testing_lookup(time_centroid, exposure, ant1, ant2,
                                          flag_row, time_bin_secs,
                                          row_meta)

    # Input rows associated with each output row
    row_idx, flag_row_idx = zip(*[(row, flag_rows) for _, _, row, flag_rows
                                  in time_bl_row_map])

    # Check that the averaged times from the test and accelerated lookup match
    assert_array_equal([t for t, _, _, _ in time_bl_row_map],
                       row_meta.time_centroid)

    avg = time_and_channel(time_centroid, exposure, ant1, ant2,
                           flag_row=flag_row,
                           time=time, interval=interval, uvw=uvw,
                           weight=weight, sigma=sigma,
                           vis=vis, flag=flag,
                           weight_spectrum=weight_spectrum,
                           sigma_spectrum=sigma_spectrum,
                           time_bin_secs=time_bin_secs,
                           chan_bin_size=chan_bin_size)

    # Take mean time, but first ant1 and ant2
    expected_time_centroids = [time[i].mean(axis=0) for i in row_idx]
    expected_times = [time[i].mean(axis=0) for i in flag_row_idx]
    expected_ant1 = [ant1[i[0]] for i in row_idx]
    expected_ant2 = [ant2[i[0]] for i in row_idx]

    # Take mean average, but sum of interval and exposure
    expected_uvw = [uvw[i].mean(axis=0) for i in row_idx]
    expected_interval = [interval[i].sum(axis=0) for i in flag_row_idx]
    expected_exposure = [exposure[i].sum(axis=0) for i in row_idx]
    expected_weight = [weight[i].mean(axis=0) for i in row_idx]
    expected_sigma = [sigma[i].mean(axis=0) for i in row_idx]

    assert_array_equal(row_meta.time_centroid, expected_time_centroids)
    assert_array_equal(row_meta.exposure, expected_exposure)
    assert_array_equal(avg.antenna1, expected_ant1)
    assert_array_equal(avg.antenna2, expected_ant2)
    assert_array_equal(avg.time, expected_times)
    assert_array_equal(avg.uvw, expected_uvw)
    assert_array_equal(avg.interval, expected_interval)
    assert_array_equal(row_meta.exposure, expected_exposure)
    assert_array_equal(avg.weight, expected_weight)
    assert_array_equal(avg.sigma, expected_sigma)

    chan_avg_shape = (row_meta.exposure.shape[0], chan_bins, flag.shape[2])

    assert avg.vis.shape == chan_avg_shape
    assert avg.flag.shape == chan_avg_shape
    assert avg.weight_spectrum.shape == chan_avg_shape
    assert avg.sigma_spectrum.shape == chan_avg_shape


@pytest.mark.parametrize("flagged_rows", [
    [], [8, 9], [4], [0, 1],
])
@pytest.mark.parametrize("time_bin_secs", [1, 2, 3, 4])
@pytest.mark.parametrize("chan_bin_size", [1, 3, 5])
def test_dask_averager(time, ant1, ant2, flagged_rows,
                       uvw, interval, weight, sigma,
                       vis, flag,
                       weight_spectrum, sigma_spectrum,
                       time_bin_secs, chan_bin_size):

    da = pytest.importorskip('dask.array')

    from africanus.averaging.dask import time_and_channel as dask_avg

    rc = (6, 4)
    fc = (4, 4, 4, 4)
    cc = (4,)

    rows = sum(rc)
    chans = sum(fc)
    corrs = sum(cc)

    time_centroid = time
    exposure = interval

    vis = vis(rows, chans, corrs)
    flag = flag(rows, chans, corrs)
    flag_row = np.zeros(time_centroid.shape[0], dtype=np.uint8)
    flag_row[flagged_rows] = 1
    flag[flagged_rows, :, :] = 1

    np_avg = time_and_channel(time_centroid, exposure, ant1, ant2,
                              flag_row=flag_row,
                              vis=vis, flag=flag,
                              time_bin_secs=time_bin_secs,
                              chan_bin_size=chan_bin_size)

    # Using chunks == shape, the dask version should match the numpy version
    da_time_centroid = da.from_array(time_centroid, chunks=rows)
    da_exposure = da.from_array(exposure, chunks=rows)
    da_flag_row = da.from_array(flag_row, chunks=rows)
    da_ant1 = da.from_array(ant1, chunks=rows)
    da_ant2 = da.from_array(ant2, chunks=rows)
    da_vis = da.from_array(vis, chunks=(rows, chans, corrs))
    da_flag = da.from_array(flag, chunks=(rows, chans, corrs))

    avg = dask_avg(da_time_centroid, da_exposure, da_ant1, da_ant2,
                   flag_row=da_flag_row,
                   vis=da_vis, flag=da_flag,
                   time_bin_secs=time_bin_secs,
                   chan_bin_size=chan_bin_size)

    assert_array_equal(np_avg.time_centroid, avg.time_centroid)
    assert_array_equal(np_avg.vis, avg.vis)

    # We can average chunked arrays too, but these will not necessarily
    # match the numpy version
    da_time_centroid = da.from_array(time_centroid, chunks=(rc,))
    da_exposure = da.from_array(exposure, chunks=(rc,))
    da_flag_row = da.from_array(flag_row, chunks=(rc,))
    da_ant1 = da.from_array(ant1, chunks=(rc,))
    da_ant2 = da.from_array(ant2, chunks=(rc,))
    da_vis = da.from_array(vis, chunks=(rc, fc, cc))
    da_flag = da.from_array(flag, chunks=(rc, fc, cc))

    avg = dask_avg(da_time_centroid, da_exposure, da_ant1, da_ant2,
                   flag_row=da_flag_row,
                   vis=da_vis, flag=da_flag,
                   time_bin_secs=time_bin_secs,
                   chan_bin_size=chan_bin_size)

    avg.vis.compute()
