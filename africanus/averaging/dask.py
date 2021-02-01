# -*- coding: utf-8 -*-


from operator import getitem

from africanus.averaging.time_and_channel_mapping import (
                row_mapper as np_row_mapper,
                channel_mapper as np_channel_mapper)
from africanus.averaging.time_and_channel_avg import (
                row_average as np_row_average,
                row_chan_average as np_row_chan_average,
                chan_average as np_chan_average,
                merge_flags as np_merge_flags,
                AVERAGING_DOCS,
                AverageOutput, ChannelAverageOutput,
                RowAverageOutput, RowChanAverageOutput)

from africanus.util.requirements import requires_optional

import numpy as np

try:
    from dask.base import tokenize
    import dask.array as da
    import dask.blockwise as db
    from dask.highlevelgraph import HighLevelGraph
except ImportError as e:
    dask_import_error = e
else:
    dask_import_error = None


def chan_metadata(row_chan_arrays, chan_arrays, chan_bin_size):
    """ Create dask array with channel metadata for each chunk channel """
    chan_chunks = None

    for array in row_chan_arrays:
        if array is not None:
            chan_chunks = array.chunks[1]
            break

    if chan_chunks is None:
        for array in chan_arrays:
            if array is not None:
                chan_chunks = array.chunks[0]
                break

    if chan_chunks is None:
        return None

    # Create a dask channel mapping structure
    name = "channel-mapper-" + tokenize(chan_chunks, chan_bin_size)
    layers = {(name, i): (np_channel_mapper, c, chan_bin_size)
              for i, c in enumerate(chan_chunks)}
    graph = HighLevelGraph.from_collections(name, layers, ())
    chunks = (chan_chunks,)
    chan_mapper = da.Array(graph, name, chunks, dtype=np.object)

    return chan_mapper


def row_mapper(time, interval, antenna1, antenna2,
               flag_row=None, time_bin_secs=1.0):
    """ Create a dask row mapping structure for each row chunk """
    return da.blockwise(np_row_mapper, ("row",),
                        time, ("row",),
                        interval, ("row",),
                        antenna1, ("row",),
                        antenna2, ("row",),
                        flag_row, None if flag_row is None else ("row",),
                        adjust_chunks={"row": lambda x: np.nan},
                        time_bin_secs=time_bin_secs,
                        meta=np.empty((0,), dtype=np.object),
                        dtype=np.object)


def _getitem_row(avg, idx, array, dims):
    """ Extract row-like arrays from a dask array of tuples """
    assert dims[0] == "row"

    name = ("row-average-getitem-%d-" % idx) + tokenize(avg, idx)
    layers = db.blockwise(getitem, name, dims,
                          avg.name, ("row",),
                          idx, None,
                          new_axes=dict(zip(dims[1:], array.shape[1:])),
                          numblocks={avg.name: avg.numblocks})
    graph = HighLevelGraph.from_collections(name, layers, (avg,))
    chunks = avg.chunks + tuple((s,) for s in array.shape[1:])

    return da.Array(graph, name, chunks,
                    meta=np.empty((0,)*len(dims), dtype=array.dtype),
                    dtype=array.dtype)


def _row_average_wrapper(row_meta, ant1, ant2, flag_row,
                         time_centroid, exposure, uvw,
                         weight, sigma):
    return np_row_average(row_meta, ant1, ant2, flag_row,
                          time_centroid, exposure,
                          uvw[0] if uvw is not None else None,
                          weight[0] if weight is not None else None,
                          sigma[0] if sigma is not None else None)


def row_average(row_meta, ant1, ant2, flag_row=None,
                time_centroid=None, exposure=None, uvw=None,
                weight=None, sigma=None):
    """ Average row-based dask arrays """

    rd = ("row",)
    rcd = ("row", "corr")

    # (output, array, dims)
    args = [(False, row_meta, rd),
            (True, ant1, rd),
            (True, ant2, rd),
            (False, flag_row, None if flag_row is None else rd),
            (True, time_centroid, None if time_centroid is None else rd),
            (True, exposure, None if exposure is None else rd),
            (True, uvw, None if uvw is None else ("row", "uvw")),
            (True, weight, None if weight is None else rcd),
            (True, sigma, None if sigma is None else rcd)]

    avg = da.blockwise(_row_average_wrapper, rd,
                       *(v for pair in args for v in pair[1:]),
                       align_arrays=False,
                       adjust_chunks={"row": lambda x: np.nan},
                       meta=np.empty((0,)*len(rd), dtype=np.object),
                       dtype=np.object)

    # ant1, ant2, time_centroid, exposure, uvw, weight, sigma
    out_args = [(a, dims) for out, a, dims in args if out is True]

    tuple_gets = [None if a is None else _getitem_row(avg, i, a, dims)
                  for i, (a, dims) in enumerate(out_args)]

    return RowAverageOutput(*tuple_gets)


def _getitem_row_chan(avg, idx, dtype):
    """ Extract (row,chan,corr) arrays from dask array of tuples """
    name = ("row-chan-average-getitem-%d-" % idx) + tokenize(avg, idx)
    dim = ("row", "chan", "corr")

    layers = db.blockwise(getitem, name, dim,
                          avg.name, dim,
                          idx, None,
                          numblocks={avg.name: avg.numblocks})

    graph = HighLevelGraph.from_collections(name, layers, (avg,))
    return da.Array(graph, name, avg.chunks,
                    meta=np.empty((0,)*len(dim), dtype=np.object),
                    dtype=dtype)


_row_chan_avg_dims = ("row", "chan", "corr")


def row_chan_average(row_meta, chan_meta, flag_row=None, weight=None,
                     vis=None, flag=None,
                     weight_spectrum=None, sigma_spectrum=None,
                     chan_bin_size=1):
    """ Average (row,chan,corr)-based dask arrays """

    if chan_meta is None:
        return RowChanAverageOutput(None, None, None, None)

    # We don't know how many rows are in each row chunk,
    # but we can simply divide each channel chunk size by the bin size
    adjust_chunks = {
        "row": lambda r: np.nan,
        "chan": lambda c: (c + chan_bin_size - 1) // chan_bin_size
    }

    flag_row_dims = None if flag_row is None else ("row",)
    weight_dims = None if weight is None else ("row", "corr")
    vis_dims = None if vis is None else _row_chan_avg_dims
    flag_dims = None if flag is None else _row_chan_avg_dims
    ws_dims = None if weight_spectrum is None else _row_chan_avg_dims
    ss_dims = None if sigma_spectrum is None else _row_chan_avg_dims

    avg = da.blockwise(np_row_chan_average, _row_chan_avg_dims,
                       row_meta, ("row",),
                       chan_meta, ("chan",),
                       flag_row, flag_row_dims,
                       weight, weight_dims,
                       vis, vis_dims,
                       flag, flag_dims,
                       weight_spectrum, ws_dims,
                       sigma_spectrum, ss_dims,
                       align_arrays=False,
                       adjust_chunks=adjust_chunks,
                       meta=np.empty((0,)*len(_row_chan_avg_dims),
                                     dtype=np.object),
                       dtype=np.object)

    tuple_gets = (None if a is None else _getitem_row_chan(avg, i, a.dtype)
                  for i, a in enumerate([vis, flag,
                                         weight_spectrum,
                                         sigma_spectrum]))

    return RowChanAverageOutput(*tuple_gets)


def _getitem_chan(avg, idx, dtype):
    """ Extract row-like arrays from a dask array of tuples """
    name = ("chan-average-getitem-%d-" % idx) + tokenize(avg, idx)
    layers = db.blockwise(getitem, name, ("chan",),
                          avg.name, ("chan",),
                          idx, None,
                          numblocks={avg.name: avg.numblocks})
    graph = HighLevelGraph.from_collections(name, layers, (avg,))
    return da.Array(graph, name, avg.chunks,
                    meta=np.empty((0,), dtype=dtype),
                    dtype=dtype)


def chan_average(chan_meta, chan_freq=None, chan_width=None,
                 effective_bw=None, resolution=None, chan_bin_size=1):

    if chan_meta is None:
        return ChannelAverageOutput(None, None)

    adjust_chunks = {
        "chan": lambda c: (c + chan_bin_size - 1) // chan_bin_size
    }

    cdim = ("chan",)

    avg = da.blockwise(np_chan_average, cdim,
                       chan_meta, cdim,
                       chan_freq, None if chan_freq is None else cdim,
                       chan_width, None if chan_width is None else cdim,
                       effective_bw, None if effective_bw is None else cdim,
                       resolution, None if resolution is None else cdim,
                       adjust_chunks=adjust_chunks,
                       meta=np.empty((0,), dtype=np.object),
                       dtype=np.object)

    tuple_gets = (None if a is None else _getitem_chan(avg, i, a.dtype)
                  for i, a in enumerate([chan_freq, chan_width,
                                         effective_bw, resolution]))

    return ChannelAverageOutput(*tuple_gets)


def merge_flags(flag_row, flag):
    """ Perform flag merging on dask arrays """
    if flag_row is None and flag is not None:
        return da.blockwise(np_merge_flags, "r",
                            flag_row, None,
                            flag, "rfc",
                            concatenate=True,
                            dtype=flag.dtype)
    elif flag_row is not None and flag is None:
        return da.blockwise(np_merge_flags, "r",
                            flag_row, "r",
                            None, None,
                            dtype=flag_row.dtype)
    elif flag_row is not None and flag is not None:
        return da.blockwise(np_merge_flags, "r",
                            flag_row, "r",
                            flag, "rfc",
                            concatenate=True,
                            dtype=flag_row.dtype)
    else:
        return None


@requires_optional("dask.array", dask_import_error)
def time_and_channel(time, interval, antenna1, antenna2,
                     time_centroid=None, exposure=None, flag_row=None,
                     uvw=None, weight=None, sigma=None,
                     chan_freq=None, chan_width=None,
                     effective_bw=None, resolution=None,
                     vis=None, flag=None,
                     weight_spectrum=None, sigma_spectrum=None,
                     time_bin_secs=1.0, chan_bin_size=1):

    row_chan_arrays = (vis, flag, weight_spectrum, sigma_spectrum)
    chan_arrays = (chan_freq, chan_width, effective_bw, resolution)

    # The flow of this function should match that of the numba
    # time_and_channel implementation

    # Merge flag_row and flag arrays
    flag_row = merge_flags(flag_row, flag)

    # Generate row mapping metadata
    row_meta = row_mapper(time, interval,
                          antenna1, antenna2,
                          flag_row=flag_row,
                          time_bin_secs=time_bin_secs)

    # Generate channel mapping metadata
    chan_meta = chan_metadata(row_chan_arrays, chan_arrays, chan_bin_size)

    # Average row data
    row_data = row_average(row_meta, antenna1, antenna2,
                           flag_row=flag_row,
                           time_centroid=time_centroid,
                           exposure=exposure, uvw=uvw,
                           weight=weight, sigma=sigma)

    # Average channel data
    row_chan_data = row_chan_average(row_meta, chan_meta,
                                     flag_row=flag_row, weight=weight,
                                     vis=vis, flag=flag,
                                     weight_spectrum=weight_spectrum,
                                     sigma_spectrum=sigma_spectrum,
                                     chan_bin_size=chan_bin_size)

    chan_data = chan_average(chan_meta,
                             chan_freq=chan_freq,
                             chan_width=chan_width,
                             effective_bw=effective_bw,
                             resolution=resolution)

    # Merge output tuples
    return AverageOutput(_getitem_row(row_meta, 1, time, ("row",)),
                         _getitem_row(row_meta, 2, interval, ("row",)),
                         (_getitem_row(row_meta, 3, flag_row, ("row",))
                          if flag_row is not None else None),
                         row_data.antenna1,
                         row_data.antenna2,
                         row_data.time_centroid,
                         row_data.exposure,
                         row_data.uvw,
                         row_data.weight,
                         row_data.sigma,
                         chan_data.chan_freq,
                         chan_data.chan_width,
                         chan_data.effective_bw,
                         chan_data.resolution,
                         row_chan_data.vis,
                         row_chan_data.flag,
                         row_chan_data.weight_spectrum,
                         row_chan_data.sigma_spectrum)


try:
    time_and_channel.__doc__ = AVERAGING_DOCS.substitute(
                                    array_type=":class:`dask.array.Array`")
except AttributeError:
    pass
