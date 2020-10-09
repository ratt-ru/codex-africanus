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

    vis = vis(time.shape[0], nchan, ncorr)
    flag = flag(time.shape[0], nchan, ncorr)
    flag_row = np.all(flag, axis=(1, 2))
    weight_spectrum = np.random.random(size=flag.shape).astype(np.float64)
    sigma_spectrum = np.random.random(size=flag.shape).astype(np.float64)

    import time as timing

    none_flag_row = None
    start = timing.perf_counter()
    meta = atemkeng_mapper(time, interval, ant1, ant2, uvw,
                           chan_width, chan_freq,
                           max_uvw_dist,
                           flag_row=none_flag_row, max_fov=3.0,
                           decorrelation=decorrelation)

    time_centroid = time
    exposure = interval

    start = timing.perf_counter()
    row_avg = row_average(meta, ant1, ant2, none_flag_row,  # noqa: F841
                          time_centroid, exposure,
                          uvw, weight=None, sigma=None)

    print("row_average: %f" % (timing.perf_counter() - start))

    assert_array_almost_equal(row_avg.exposure, meta.interval)
    assert_array_almost_equal(row_avg.time_centroid, meta.time)

    # vis = vis(time.shape[0], nchan, ncorr)
    # flag = flag(time.shape[0], nchan, ncorr)
    # weight_spectrum = np.random.random(size=flag.shape).astype(np.float64)
    # sigma_spectrum = np.random.random(size=flag.shape).astype(np.float64)

    start = timing.perf_counter()
    row_chan = row_chan_average(meta,  # noqa: F841
                                flag_row=flag_row,
                                visibilities=vis, flag=flag,
                                weight_spectrum=weight_spectrum,
                                sigma_spectrum=sigma_spectrum)

    row_chan2 = row_chan_average(meta,  # noqa: F841
                                 flag_row=flag_row,
                                 visibilities=(vis, vis), flag=flag,
                                 weight_spectrum=weight_spectrum,
                                 sigma_spectrum=sigma_spectrum)

    assert_array_almost_equal(row_chan.vis, row_chan2.vis[0])
    assert_array_almost_equal(row_chan.vis, row_chan2.vis[1])
    assert_array_almost_equal(row_chan.flag, row_chan2.flag)
    assert_array_almost_equal(row_chan.weight_spectrum,
                              row_chan2.weight_spectrum)
    assert_array_almost_equal(row_chan.sigma_spectrum,
                              row_chan2.sigma_spectrum)

    print("row_chan_average: %f" % (timing.perf_counter() - start))

    print(vis.shape, vis.nbytes / (1024.**2),
          row_chan.vis.shape, row_chan.vis.nbytes / (1024.**2))

    avg = bda(time, interval, ant1, ant2,  # noqa: F841
              time_centroid=time_centroid, exposure=exposure,
              flag_row=flag_row, uvw=uvw,
              chan_freq=chan_freq, chan_width=chan_width,
              visibilities=vis, flag=flag,
              weight_spectrum=weight_spectrum,
              sigma_spectrum=sigma_spectrum,
              max_uvw_dist=max_uvw_dist)


@pytest.mark.parametrize("vis_format", ["ragged", "flat"])
def test_dask_bda_avg(time, interval, ants,   # noqa: F811
                      phase_dir,              # noqa: F811
                      chan_freq, chan_width,  # noqa: F811
                      vis, flag,              # noqa: F811
                      vis_format):            # noqa: F811
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
    flag_row = np.zeros(time.shape[0], dtype=np.uint8)
    vis = vis(time.shape[0], nchan, ncorr)
    flag = flag(time.shape[0], nchan, ncorr)
    mask = flag_row.astype(np.bool)
    flag[mask, :, :] = 1

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
                   visibilities=da_vis, flag=da_flag,
                   decorrelation=decorrelation,
                   format=vis_format)

    avg = {f: getattr(avg, f) for f in ("time", "interval",
                                        "time_centroid", "exposure",
                                        "vis")}

    avg2 = dask_bda(da_time, da_interval, da_ant1, da_ant2,
                    time_centroid=da_time_centroid, exposure=da_exposure,
                    flag_row=da_flag_row, uvw=da_uvw,
                    chan_freq=da_chan_freq, chan_width=da_chan_width,
                    visibilities=(da_vis, da_vis), flag=da_flag,
                    decorrelation=decorrelation,
                    format=vis_format)

    avg2 = {f: getattr(avg2, f) for f in ("time", "interval",
                                          "time_centroid",
                                          "exposure", "vis")}

    import dask
    result = dask.persist(avg, scheduler='single-threaded')[0]
    result2 = dask.persist(avg2, scheduler='single-threaded')[0]

    assert_array_almost_equal(result['interval'], result['exposure'])
    assert_array_almost_equal(result['time'], result['time_centroid'])

    # Flatten all three visibility graphs
    dsk1 = dict(result['vis'].__dask_graph__())
    dsk2 = dict(result2['vis'][0].__dask_graph__())
    dsk3 = dict(result2['vis'][1].__dask_graph__())
    dsk2_name = result2['vis'][0].name
    dsk3_name = result2['vis'][1].name

    # For each task, compare the row dictionaries
    for k, v in dsk1.items():
        v2 = dsk2[(dsk2_name,) + k[1:]]
        v3 = dsk3[(dsk3_name,) + k[1:]]

        if vis_format == "ragged":
            assert isinstance(v, dict)
            assert isinstance(v2, dict)
            assert isinstance(v3, dict)

            # Each row in first, second and third graph match
            for rk, rv in v.items():
                assert_array_almost_equal(rv, v2[rk])
                assert_array_almost_equal(rv, v3[rk])
        elif vis_format == "flat":
            assert_array_almost_equal(v, v2)
            assert_array_almost_equal(v, v3)
        else:
            raise ValueError(f"Invalid vis_format: {vis_format}")


@pytest.fixture
def check_leaks():
    import gc
    from numba.core.runtime import rtsys

    try:
        yield None
    finally:
        gc.collect()

    stats = rtsys.get_allocation_stats()
    assert stats.alloc == stats.free
    assert stats.mi_alloc == stats.mi_free


@pytest.mark.parametrize("dtype", [np.complex64])
def test_bda_output_arrays(dtype, check_leaks):
    from africanus.averaging.bda_avg import vis_output_arrays
    from numba import njit

    @njit
    def fn(a, o):
        return vis_output_arrays(a, o)

    vis = np.random.random((10, 4, 2)).astype(dtype)
    vis2 = vis.astype(np.complex128)

    res = fn((vis, vis2), (5, 5))
    avg_vis = res[0]
    avg_weights = res[1]

    assert avg_vis[0].dtype == vis.dtype
    assert avg_vis[0].shape == (5, 5)
    assert avg_vis[1].dtype == vis2.dtype
    assert avg_vis[1].shape == (5, 5)
    assert np.all(avg_vis[0] == 0)
    assert np.all(avg_vis[1] == 0)
    assert avg_weights[0].dtype == vis.real.dtype
    assert avg_weights[0].shape == (5, 5)
    assert avg_weights[1].dtype == vis2.real.dtype
    assert avg_weights[1].shape == (5, 5)
    assert np.all(avg_weights[0] == 0)
    assert np.all(avg_weights[1] == 0)

    res = fn((vis), (5, 5))
    avg_vis = res[0]
    avg_weights = res[1]

    assert avg_vis.dtype == vis.dtype
    assert avg_vis.shape == (5, 5)
    assert avg_weights.dtype == vis.real.dtype
    assert avg_weights.shape == (5, 5)
    assert np.all(avg_vis == 0)
    assert np.all(avg_weights == 0)
