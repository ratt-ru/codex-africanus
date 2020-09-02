# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from africanus.averaging.tests.test_bda_mapping import (  # noqa: F401
                            synthesize_uvw,
                            time,
                            interval,
                            ants,
                            phase_dir,
                            chan_width,
                            chan_freq)

from africanus.averaging.bda_mapping import atemkeng_mapper
from africanus.averaging.bda_avg import row_average, row_chan_average, bda


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
                 phase_dir,              # noqa: F811
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

    decorrelation = 0.999
    max_uvw_dist = np.sqrt(np.sum(uvw**2, axis=1)).max()

    import time as timing

    start = timing.perf_counter()
    meta = atemkeng_mapper(time, interval, ant1, ant2, uvw,
                           chan_width, chan_freq,
                           max_uvw_dist,
                           flag_row=flag_row, max_fov=3.0,
                           time_bin_secs=4.0,
                           decorrelation=decorrelation)

    print("mapping: %f" % (timing.perf_counter() - start))

    time_centroid = time
    exposure = interval

    start = timing.perf_counter()
    row_avg = row_average(meta, ant1, ant2, flag_row,  # noqa: F841
                          time_centroid, exposure,
                          uvw, weight=None, sigma=None)

    print("row_average: %f" % (timing.perf_counter() - start))

    vis = vis(time.shape[0], nchan, ncorr)
    flag = flag(time.shape[0], nchan, ncorr)
    weight_spectrum = np.random.random(size=flag.shape).astype(np.float64)
    sigma_spectrum = np.random.random(size=flag.shape).astype(np.float64)

    start = timing.perf_counter()
    row_chan = row_chan_average(meta,  # noqa: F841
                                flag_row=flag_row,
                                vis=vis, flag=flag,
                                weight_spectrum=weight_spectrum,
                                sigma_spectrum=sigma_spectrum)

    assert_array_almost_equal(row_avg.time_centroid, meta.time)
    assert_array_almost_equal(row_avg.exposure, meta.interval)

    print("row_chan_average: %f" % (timing.perf_counter() - start))

    print(vis.shape, vis.nbytes / (1024.**2),
          row_chan.vis.shape, row_chan.vis.nbytes / (1024.**2))

    avg = bda(time, interval, ant1, ant2,  # noqa: F841
              time_centroid=time_centroid, exposure=exposure,
              flag_row=flag_row, uvw=uvw,
              chan_freq=chan_freq, chan_width=chan_width,
              vis=vis, flag=flag,
              weight_spectrum=weight_spectrum,
              sigma_spectrum=sigma_spectrum,
              max_uvw_dist=max_uvw_dist)


def test_dask_bda_avg(time, interval, ants,   # noqa: F811
                      phase_dir,              # noqa: F811
                      chan_freq, chan_width,  # noqa: F811
                      vis, flag):             # noqa: F811
    da = pytest.importorskip('dask.array')
    from africanus.averaging.dask import bda as dask_bda

    time = np.unique(time)
    interval = interval[:time.shape[0]]
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
    vis = vis(time.shape[0], nchan, ncorr)
    flag = flag(time.shape[0], nchan, ncorr)

    assert time.shape == ant1.shape

    decorrelation = 0.999
    chunks = 1000

    da_time = da.from_array(time, chunks=chunks)
    da_interval = da.from_array(interval, chunks=chunks)
    da_flag_row = da.from_array(flag_row, chunks=chunks)
    da_ant1 = da.from_array(ant1, chunks=chunks)
    da_ant2 = da.from_array(ant2, chunks=chunks)
    da_uvw = da.from_array(uvw, chunks=(chunks, 3))
    da_time_centroid = da_time
    da_exposure = da_interval
    da_chan_freq = da.from_array(chan_freq, chunks=nchan)
    da_chan_width = da.from_array(chan_width, chunks=nchan)
    da_vis = da.from_array(vis, chunks=(chunks, nchan, ncorr))
    da_flag = da.from_array(flag, chunks=(chunks, nchan, ncorr))

    avg = dask_bda(da_time, da_interval, da_ant1, da_ant2,
                   time_centroid=da_time_centroid, exposure=da_exposure,
                   flag_row=da_flag_row, uvw=da_uvw,
                   chan_freq=da_chan_freq, chan_width=da_chan_width,
                   vis=da_vis, flag=da_flag,
                   decorrelation=decorrelation,
                   format="ragged")

    avg = {f: getattr(avg, f) for f in ("time", "interval", "vis")}

    import dask
    result = dask.persist(avg, scheduler='single-threaded')[0]

    from pprint import pprint
    pprint({k: v.shape for k, v in result.items() if v is not None})
