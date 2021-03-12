# -*- coding: utf-8 -*-

from collections import defaultdict

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest

from africanus.averaging.bda_mapping import RowMapOutput
from africanus.averaging.bda_avg import row_average, row_chan_average
from africanus.averaging.dask import bda as dask_bda


@pytest.fixture(params=[
    # 5 rows, 4 channels => 3 rows
    #
    # row 1 contains 2 channels
    # row 2 contains 3 channels
    # row 3 contains 1 channel
    [[0, 0, 1, 1],
     [0, 0, 1, 1],
     [2, 3, 3, 4],
     [2, 3, 3, 4],
     [5, 5, 5, 5]]
])
def bda_test_map(request):
    return np.asarray(request.param)


@pytest.fixture(params=[
    # No flags
    [[0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0]],

    # Row 0 and 1 flagged
    [[1, 1, 1, 1],
     [1, 1, 1, 1],
     [0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0]],

    # Row 2 flagged
    [[0, 0, 0, 0],
     [0, 0, 0, 0],
     [1, 1, 1, 1],
     [0, 0, 0, 0],
     [0, 0, 0, 0]],

    # Row 0, 2, 4 flagged
    [[1, 1, 1, 1],
     [0, 0, 0, 0],
     [1, 1, 1, 1],
     [0, 0, 0, 0],
     [1, 1, 1, 1]],


    # All flagged
    [[1, 1, 1, 1],
     [1, 1, 1, 1],
     [1, 1, 1, 1],
     [1, 1, 1, 1],
     [1, 1, 1, 1]],


    # Partially flagged
    [[0, 1, 0, 1],
     [0, 1, 0, 0],
     [0, 0, 0, 0],
     [1, 1, 1, 1],
     [1, 0, 0, 0]],
])
def flags(request):
    return np.asarray(request.param)


@pytest.fixture
def inv_bda_test_map(bda_test_map):
    """ Generates a :code:`{out_row: [in_row, in_chan]}` mapping"""
    inv = defaultdict(list)
    for idx in np.ndindex(*bda_test_map.shape):
        inv[bda_test_map[idx]].append(idx)

    return {ro: tuple(list(i) for i in zip(*v))
            for ro, v in inv.items()}


def _effective_row_map(flag_row, inv_row_map):
    """ Build an effective row map """
    emap = []

    for _, (rows, counts) in sorted(inv_row_map.items()):
        if flag_row[rows].all():
            # Pass through all rows if the entire bin is flagged
            emap.append((rows, counts))
        else:
            # Pass through only unflagged rows if some of the bin is flagged
            it = ((r, c) for r, c in zip(rows, counts) if flag_row[r] == 0)
            emap.append(tuple(map(list, zip(*it))))

    return emap


def _effective_rowchan_map(flags, inv_bda_test_map):
    emap = []

    for _, (rows, chans) in sorted(inv_bda_test_map.items()):
        if flags[rows, chans].all():
            emap.append((rows, chans))
        else:
            it = ((r, c) for r, c in zip(rows, chans) if flags[r, c] == 0)
            emap.append(tuple(map(list, zip(*it))))

    return emap


def _calc_sigma(weight, sigma, rows):
    weight = weight[rows]
    sigma = sigma[rows]
    numerator = (sigma**2 * weight**2).sum(axis=0)
    denominator = weight.sum(axis=0)**2
    denominator[denominator == 0.0] = 1.0

    return np.sqrt(numerator / denominator)


def test_bda_avg(bda_test_map, inv_bda_test_map, flags):
    rs = np.random.RandomState(42)

    # Derive flag_row from flags
    flag_row = flags.all(axis=1)

    out_chan = np.array([np.unique(rows).size for rows
                         in np.unique(bda_test_map, axis=0)])

    # Number of output rows is sum of row channels
    out_row = out_chan.sum()
    assert out_row == bda_test_map.max() + 1

    in_row, in_chan = bda_test_map.shape
    in_corr = 4
    offsets = np.array([0, 2, 5, out_row])
    assert_array_equal(offsets[:-1], np.unique(bda_test_map[:, 0]))

    time = np.linspace(1.0, float(in_row), in_row, dtype=np.float64)  # noqa
    interval = np.full(in_row, 1.0, dtype=np.float64)  # noqa
    uvw = np.arange(in_row*3).reshape(in_row, 3).astype(np.float64)
    weight = rs.normal(size=(in_row, in_corr))
    sigma = rs.normal(size=(in_row, in_corr))
    chan_width = np.repeat(.856e9 / out_chan, out_chan)

    # Aggregate time and interval, in_row => out_row
    # first channel in the map. We're only averaging over
    # row so we don't want to aggregate per channel
    idx = bda_test_map[np.arange(in_row), 0]
    out_time = np.zeros(out_row, dtype=time.dtype)
    out_counts = np.zeros(out_row, dtype=np.uint32)
    out_interval = np.zeros(out_row, dtype=interval.dtype)
    np.add.at(out_time, idx, time)
    np.add.at(out_counts, idx, 1)
    np.add.at(out_interval, idx, interval)

    # Now copy values to other channel positions
    # and normalise time values
    copy_idx = np.repeat(offsets[:-1], np.diff(offsets))
    out_time[:] = out_time[copy_idx]
    out_counts[:] = out_counts[copy_idx]
    out_interval[:] = out_interval[copy_idx]
    out_time /= out_counts

    inv_row_map = {ro: np.unique(rows, return_counts=True)
                   for ro, (rows, _) in inv_bda_test_map.items()}
    out_time2 = [time[rows].sum() / len(counts) for _, (rows, counts)
                 in sorted(inv_row_map.items())]
    assert_array_equal(out_time, out_time2)

    out_interval2 = [interval[rows].sum() for _, (rows, _)
                     in sorted(inv_row_map.items())]
    assert_array_equal(out_interval, out_interval2)

    out_flag_row = [flag_row[rows].all() for _, (rows, _)
                    in sorted(inv_row_map.items())]

    meta = RowMapOutput(bda_test_map, offsets,
                        chan_width, out_time, out_interval,
                        None, out_flag_row)

    ant1 = np.full(in_row, 0, dtype=np.int32)
    ant2 = np.full(in_row, 1, dtype=np.int32)

    row_avg = row_average(meta, ant1, ant2,
                          time_centroid=time,
                          exposure=interval,
                          uvw=uvw,
                          weight=weight, sigma=sigma,
                          flag_row=flag_row)

    assert_array_equal(row_avg.antenna1, 0)
    assert_array_equal(row_avg.antenna2, 1)

    # Effective averages
    effective_map = _effective_row_map(flag_row, inv_row_map)
    out_time_centroid = [time[r].sum() / len(c) for r, c in effective_map]
    out_interval = [interval[r].sum() for r, _ in effective_map]
    out_uvw = [uvw[r].sum(axis=0) / len(c) for r, c in effective_map]
    out_weight = [weight[r].sum(axis=0) for r, _ in effective_map]
    out_sigma = [_calc_sigma(weight, sigma, r) for r, _ in effective_map]

    assert_array_equal(row_avg.time_centroid, out_time_centroid)
    assert_array_equal(row_avg.exposure, out_interval)
    assert_array_equal(row_avg.uvw, out_uvw)
    assert_array_equal(row_avg.weight, out_weight)
    assert_array_equal(row_avg.sigma, out_sigma)

    vshape = (in_row, in_chan, in_corr)
    vis = rs.normal(size=vshape) + rs.normal(size=vshape)*1j
    weight_spectrum = rs.normal(size=vshape)
    sigma_spectrum = rs.normal(size=vshape)
    flag = np.broadcast_to(flags[:, :, None], vshape)

    effective_map = _effective_rowchan_map(flags, inv_bda_test_map)
    out_ws = np.stack([
        weight_spectrum[r, c, :].sum(axis=0) for r, c in effective_map])
    out_ss = np.stack([
        (sigma_spectrum[r, c, :]**2 * weight_spectrum[r, c, :]**2).sum(axis=0)
        for r, c in effective_map])
    out_vis = np.stack([
        (vis[r, c, :]*weight_spectrum[r, c, :]).sum(axis=0)
        for r, c in effective_map])
    out_flag = np.stack([flag[r, c, :].all(axis=0) for r, c in effective_map])

    weight_div = out_ws.copy()
    weight_div[weight_div == 0.0] = 1.0
    out_vis /= weight_div
    out_ss = np.sqrt(out_ss / (weight_div**2))

    # Broadcast flag data up to correlation dimension
    row_chan_avg = row_chan_average(
        meta, flag_row=flag_row, visibilities=vis,
        weight_spectrum=weight_spectrum,
        sigma_spectrum=sigma_spectrum,
        flag=flag)

    assert_array_almost_equal(row_chan_avg.visibilities, out_vis)
    assert_array_almost_equal(row_chan_avg.flag, out_flag)
    assert_array_almost_equal(row_chan_avg.weight_spectrum, out_ws)
    assert_array_almost_equal(row_chan_avg.sigma_spectrum, out_ss)


@pytest.mark.parametrize("vis_format", ["ragged", "flat"])
def test_dask_bda_avg(vis_format):
    da = pytest.importorskip('dask.array')

    dim_chunks = {
        "chan": (4,),
        "time": (5, 4, 5),
        "ant": (7,),
        "corr": (4,)
    }

    ant1, ant2 = np.triu_indices(sum(dim_chunks["ant"]), 1)
    ant1 = ant1.astype(np.int32)
    ant2 = ant2.astype(np.int32)
    nbl = ant1.shape[0]
    ntime = sum(dim_chunks["time"])
    time = np.linspace(1.0, 2.0, ntime)
    time = np.repeat(time, nbl)
    ant1 = np.tile(ant1, ntime)
    ant2 = np.tile(ant2, ntime)
    interval = np.full(time.shape, 1.0)

    row_chunks = tuple(t*nbl for t in dim_chunks["time"])
    nrow = sum(row_chunks)
    assert nrow == time.shape[0]

    nchan = sum(dim_chunks["chan"])
    ncorr = sum(dim_chunks["corr"])

    flag_row = np.zeros(time.shape[0], dtype=np.uint8)
    vshape = (nrow, nchan, ncorr)

    rs = np.random.RandomState(42)
    uvw = rs.normal(size=(nrow, 3))
    vis = rs.normal(size=vshape) + rs.normal(size=vshape)*1j
    flag = rs.randint(0, 2, size=vshape)
    flag_row = flag.all(axis=(1, 2))
    assert flag_row.shape == (nrow,)

    chan_freq = np.linspace(.856e9, 2*.856e9, nchan)
    chan_width = np.full(nchan, .856e9 / nchan)
    chan_chunks = dim_chunks["chan"]

    decorrelation = 0.999

    vis_chunks = (row_chunks, chan_chunks, dim_chunks["corr"])

    da_time = da.from_array(time, chunks=row_chunks)
    da_interval = da.from_array(interval, chunks=row_chunks)
    da_flag_row = da.from_array(flag_row, chunks=row_chunks)
    da_ant1 = da.from_array(ant1, chunks=row_chunks)
    da_ant2 = da.from_array(ant2, chunks=row_chunks)
    da_uvw = da.from_array(uvw, chunks=(row_chunks, 3))
    da_time_centroid = da_time
    da_exposure = da_interval
    da_chan_freq = da.from_array(chan_freq, chunks=chan_chunks)
    da_chan_width = da.from_array(chan_width, chunks=chan_chunks)
    da_vis = da.from_array(vis, chunks=vis_chunks)
    da_flag = da.from_array(flag, chunks=vis_chunks)

    avg = dask_bda(da_time, da_interval, da_ant1, da_ant2,
                   time_centroid=da_time_centroid, exposure=da_exposure,
                   flag_row=da_flag_row, uvw=da_uvw,
                   chan_freq=da_chan_freq, chan_width=da_chan_width,
                   visibilities=da_vis, flag=da_flag,
                   decorrelation=decorrelation,
                   format=vis_format)

    avg = {f: getattr(avg, f) for f in ("time", "interval",
                                        "time_centroid", "exposure",
                                        "visibilities")}

    avg2 = dask_bda(da_time, da_interval, da_ant1, da_ant2,
                    time_centroid=da_time_centroid, exposure=da_exposure,
                    flag_row=da_flag_row, uvw=da_uvw,
                    chan_freq=da_chan_freq, chan_width=da_chan_width,
                    visibilities=(da_vis, da_vis), flag=da_flag,
                    decorrelation=decorrelation,
                    format=vis_format)

    avg2 = {f: getattr(avg2, f) for f in ("time", "interval",
                                          "time_centroid", "exposure",
                                          "visibilities")}

    import dask
    result = dask.persist(avg, scheduler='single-threaded')[0]
    result2 = dask.persist(avg2, scheduler='single-threaded')[0]

    assert_array_almost_equal(result['interval'], result['exposure'])
    assert_array_almost_equal(result['time'], result['time_centroid'])

    # Flatten all three visibility graphs
    dsk1 = dict(result["visibilities"].__dask_graph__())
    dsk2 = dict(result2["visibilities"][0].__dask_graph__())
    dsk3 = dict(result2["visibilities"][1].__dask_graph__())
    dsk2_name = result2["visibilities"][0].name
    dsk3_name = result2["visibilities"][1].name

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
