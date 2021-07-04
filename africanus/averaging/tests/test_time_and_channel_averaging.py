# -*- coding: utf-8 -*-


import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
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
def test_averager(time, ant1, ant2, flagged_rows,
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
    flag[flag_row.astype(np.bool_), :, :] = 1
    flag[~flag_row.astype(np.bool_), :, :] = 0
    assert_array_equal(flag.all(axis=(1, 2)).astype(np.uint8), flag_row)

    row_meta = row_mapper(time, interval, ant1, ant2, flag_row, time_bin_secs)
    chan_map, chan_bins = channel_mapper(nchan, chan_bin_size)

    time_bl_row_map = _gen_testing_lookup(time_centroid, exposure, ant1, ant2,
                                          flag_row, time_bin_secs,
                                          row_meta)

    # Effective and Nominal rows associated with each output row
    eff_idx, nom_idx = zip(*[(nrows, erows) for _, _, nrows, erows
                             in time_bl_row_map])

    eff_idx = [ei for ei in eff_idx if len(ei) > 0]

    # Check that the averaged times from the test and accelerated lookup match
    assert_array_equal([t for t, _, _, _ in time_bl_row_map],
                       row_meta.time)

    avg = time_and_channel(time, interval, ant1, ant2,
                           flag_row=flag_row,
                           time_centroid=time, exposure=exposure, uvw=uvw,
                           weight=weight, sigma=sigma,
                           chan_freq=frequency, chan_width=chan_width,
                           visibilities=vis, flag=flag,
                           weight_spectrum=weight_spectrum,
                           sigma_spectrum=sigma_spectrum,
                           time_bin_secs=time_bin_secs,
                           chan_bin_size=chan_bin_size)

    # Take mean time, but first ant1 and ant2
    expected_time_centroids = [time_centroid[i].mean(axis=0) for i in eff_idx]
    expected_times = [time[i].mean(axis=0) for i in nom_idx]
    expected_ant1 = [ant1[i[0]] for i in nom_idx]
    expected_ant2 = [ant2[i[0]] for i in nom_idx]
    expected_flag_row = [flag_row[i].any(axis=0) for i in eff_idx]

    # Take mean average, but sum of interval and exposure
    expected_uvw = [uvw[i].mean(axis=0) for i in eff_idx]
    expected_interval = [interval[i].sum(axis=0) for i in nom_idx]
    expected_exposure = [exposure[i].sum(axis=0) for i in eff_idx]
    expected_weight = [weight[i].sum(axis=0) for i in eff_idx]
    expected_sigma = [_calc_sigma(sigma, weight, i) for i in eff_idx]

    assert_array_equal(row_meta.time, expected_times)
    assert_array_equal(row_meta.interval, expected_interval)
    assert_array_equal(row_meta.flag_row, expected_flag_row)
    assert_array_equal(avg.antenna1, expected_ant1)
    assert_array_equal(avg.antenna2, expected_ant2)
    assert_array_equal(avg.time_centroid, expected_time_centroids)
    assert_array_equal(avg.exposure, expected_exposure)
    assert_array_equal(avg.uvw, expected_uvw)
    assert_array_equal(avg.weight, expected_weight)
    assert_array_equal(avg.sigma, expected_sigma)

    chan_avg_shape = (row_meta.interval.shape[0], chan_bins, flag.shape[2])

    assert avg.visibilities.shape == chan_avg_shape
    assert avg.flag.shape == chan_avg_shape
    assert avg.weight_spectrum.shape == chan_avg_shape
    assert avg.sigma_spectrum.shape == chan_avg_shape

    chan_ranges = np.nonzero(np.ediff1d(chan_map, to_begin=1, to_end=1))[0]

    # Three python loops. Slow, but works...
    # Figure out some way to remove loops with numpy
    for orow, idx in enumerate(eff_idx):
        for ch, (cs, ce) in enumerate(zip(chan_ranges[:-1], chan_ranges[1:])):
            for corr in range(ncorr):
                in_flags = flag[idx, cs:ce, corr] != 0
                out_flag = in_flags.all()
                assert_array_equal(out_flag, avg.flag[orow, ch, corr])
                flags_match = in_flags == out_flag

                exp_vis = vis[idx, cs:ce, corr]
                exp_wts = weight_spectrum[idx, cs:ce, corr]
                exp_sigma = sigma_spectrum[idx, cs:ce, corr]

                # Use matching to flags to decide which
                # samples contribute to the bin
                chunk_exp_vis = exp_vis[flags_match]
                chunk_exp_wts = exp_wts[flags_match]
                chunk_exp_sigma = exp_sigma[flags_match]

                exp_vis = (chunk_exp_vis * chunk_exp_wts).sum()
                exp_sigma = (chunk_exp_sigma**2 * chunk_exp_wts**2).sum()
                exp_wts = chunk_exp_wts.sum()

                if exp_wts != 0.0:
                    exp_vis = exp_vis / exp_wts
                    exp_sigma = np.sqrt(exp_sigma / (exp_wts**2))

                assert_array_almost_equal(exp_vis,
                                          avg.visibilities[orow, ch, corr])
                assert_array_almost_equal(exp_wts,
                                          avg.weight_spectrum[orow, ch, corr])
                assert_array_almost_equal(exp_sigma,
                                          avg.sigma_spectrum[orow, ch, corr])


@pytest.mark.parametrize("flagged_rows", [
    [], [8, 9], [4], [0, 1],
])
@pytest.mark.parametrize("time_bin_secs", [1, 2, 3, 4])
@pytest.mark.parametrize("chan_bin_size", [1, 3, 5])
def test_dask_averager(time, ant1, ant2, flagged_rows,
                       uvw, interval, weight, sigma,
                       frequency, chan_width,
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
    flag[flag_row.astype(np.bool_), :, :] = 1
    flag[~flag_row.astype(np.bool_), :, :] = 0

    np_avg = time_and_channel(time_centroid, exposure, ant1, ant2,
                              flag_row=flag_row,
                              chan_freq=frequency, chan_width=chan_width,
                              effective_bw=chan_width, resolution=chan_width,
                              visibilities=vis, flag=flag,
                              sigma_spectrum=sigma_spectrum,
                              weight_spectrum=weight_spectrum,
                              time_bin_secs=time_bin_secs,
                              chan_bin_size=chan_bin_size)

    # Using chunks == shape, the dask version should match the numpy version
    da_time_centroid = da.from_array(time_centroid, chunks=rows)
    da_exposure = da.from_array(exposure, chunks=rows)
    da_flag_row = da.from_array(flag_row, chunks=rows)
    da_weight = da.from_array(weight, chunks=(rows, corrs))
    da_sigma = da.from_array(sigma, chunks=(rows, corrs))
    da_ant1 = da.from_array(ant1, chunks=rows)
    da_ant2 = da.from_array(ant2, chunks=rows)
    da_chan_freq = da.from_array(frequency, chunks=chans)
    da_chan_width = da.from_array(chan_width, chunks=chans)
    da_weight_spectrum = da.from_array(weight_spectrum,
                                       chunks=(rows, chans, corrs))
    da_sigma_spectrum = da.from_array(sigma_spectrum,
                                      chunks=(rows, chans, corrs))
    da_vis = da.from_array(vis, chunks=(rows, chans, corrs))
    da_flag = da.from_array(flag, chunks=(rows, chans, corrs))

    avg = dask_avg(da_time_centroid, da_exposure, da_ant1, da_ant2,
                   flag_row=da_flag_row,
                   chan_freq=da_chan_freq, chan_width=da_chan_width,
                   effective_bw=da_chan_width, resolution=da_chan_width,
                   weight=da_weight, sigma=da_sigma,
                   visibilities=da_vis, flag=da_flag,
                   weight_spectrum=da_weight_spectrum,
                   sigma_spectrum=da_sigma_spectrum,
                   time_bin_secs=time_bin_secs,
                   chan_bin_size=chan_bin_size)

    # Compute all the averages in one go
    (avg_time_centroid, avg_exposure, avg_flag_row,
     avg_chan_freq, avg_chan_width,
     avg_resolution, avg_vis, avg_flag) = da.compute(
                              avg.time_centroid,
                              avg.exposure,
                              avg.flag_row,
                              avg.chan_freq,
                              avg.chan_width,
                              avg.resolution,
                              avg.visibilities, avg.flag)

    # Should match
    assert_array_equal(np_avg.time_centroid, avg_time_centroid)
    assert_array_equal(np_avg.exposure, avg_exposure)
    assert_array_equal(np_avg.flag_row, avg_flag_row)
    assert_array_equal(np_avg.visibilities, avg_vis)
    assert_array_equal(np_avg.flag, avg_flag)
    assert_array_equal(np_avg.chan_freq, avg_chan_freq)
    assert_array_equal(np_avg.chan_width, avg_chan_width)
    assert_array_equal(np_avg.resolution, avg_resolution)

    # We can average chunked arrays too, but these will not necessarily
    # match the numpy version
    da_time_centroid = da.from_array(time_centroid, chunks=(rc,))
    da_exposure = da.from_array(exposure, chunks=(rc,))
    da_flag_row = da.from_array(flag_row, chunks=(rc,))
    da_ant1 = da.from_array(ant1, chunks=(rc,))
    da_ant2 = da.from_array(ant2, chunks=(rc,))
    da_chan_freq = da.from_array(frequency, chunks=(fc,))
    da_chan_width = da.from_array(chan_width, chunks=(fc,))
    da_flag = da.from_array(flag, chunks=(rc, fc, cc))
    da_vis = da.from_array(vis, chunks=(rc, fc, cc))
    da_flag = da.from_array(flag, chunks=(rc, fc, cc))

    avg = dask_avg(da_time_centroid, da_exposure, da_ant1, da_ant2,
                   flag_row=da_flag_row,
                   chan_freq=da_chan_freq, chan_width=da_chan_width,
                   visibilities=da_vis, flag=da_flag,
                   time_bin_secs=time_bin_secs,
                   chan_bin_size=chan_bin_size)

    # Compute all the fields
    fields = [getattr(avg, f) for f in avg._fields]
    avg = type(avg)(*da.compute(fields)[0])

    # Get same result with a visibility tuple
    avg2 = dask_avg(da_time_centroid, da_exposure, da_ant1, da_ant2,
                    flag_row=da_flag_row,
                    chan_freq=da_chan_freq, chan_width=da_chan_width,
                    visibilities=(da_vis, da_vis), flag=da_flag,
                    time_bin_secs=time_bin_secs,
                    chan_bin_size=chan_bin_size)

    assert_array_equal(avg.visibilities, avg2.visibilities[0])
    assert_array_equal(avg.visibilities, avg2.visibilities[1])
