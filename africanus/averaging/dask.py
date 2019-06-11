# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from operator import getitem

from africanus.averaging.time_and_channel_mapping import (row_mapper,
                                                          channel_mapper)
from africanus.averaging.time_and_channel_avg import (row_average,
                                                      row_chan_average,
                                                      chan_average,
                                                      merge_flags,
                                                      AVERAGING_DOCS,
                                                      AverageOutput,
                                                      ChannelAverageOutput,
                                                      RowAverageOutput,
                                                      RowChanAverageOutput)
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


def _row_chan_metadata(arrays, chan_bin_size):
    """ Create dask array with channel metadata for each chunk channel """
    for array in arrays:
        if array is None:
            continue

        # Create a dask channel mapping structure
        name = "channel-mapper-" + tokenize(array.chunks[1], chan_bin_size)
        layers = {(name, i): (channel_mapper, c, chan_bin_size)
                  for i, c in enumerate(array.chunks[1])}
        graph = HighLevelGraph.from_collections(name, layers, ())
        chunks = (array.chunks[1],)
        chan_mapper = da.Array(graph, name, chunks, dtype=np.object)

        return chan_mapper

    return None


def _dask_row_mapper(time, interval, antenna1, antenna2,
                     flag_row=None, time_bin_secs=1.0):
    """ Create a dask row mapping structure for each row chunk """
    return da.blockwise(row_mapper, ("row",),
                        time, ("row",),
                        interval, ("row",),
                        antenna1, ("row",),
                        antenna2, ("row",),
                        flag_row, None if flag_row is None else ("row",),
                        time_bin_secs=time_bin_secs,
                        dtype=np.object)


def _getitem_row(avg, idx, dtype):
    """ Extract row-like arrays from a dask array of tuples """
    name = ("row-average-getitem-%d-" % idx) + tokenize(avg, idx)
    layers = db.blockwise(getitem, name, ("row",),
                          avg.name, ("row",),
                          idx, None,
                          numblocks={avg.name: avg.numblocks})
    graph = HighLevelGraph.from_collections(name, layers, (avg,))
    return da.Array(graph, name, avg.chunks, dtype=dtype)


def _dask_row_average(row_meta, ant1, ant2, flag_row=None,
                      time_centroid=None, exposure=None, uvw=None,
                      weight=None, sigma=None):
    """ Average row-based dask arrays """

    rd = ("row",)
    rcd = ("row", "corr")

    avg = da.blockwise(row_average, rd,
                       row_meta, rd,
                       ant1, rd,
                       ant2, rd,
                       flag_row, None if flag_row is None else rd,
                       time_centroid, None if time_centroid is None else rd,
                       exposure, None if exposure is None else rd,
                       uvw, None if uvw is None else ("row", "3"),
                       weight, None if weight is None else rcd,
                       sigma, None if sigma is None else rcd,
                       adjust_chunks={"row": lambda x: np.nan},
                       dtype=np.object)

    tuple_gets = (None if a is None else _getitem_row(avg, i, a.dtype)
                  for i, a in enumerate([ant1, ant2, time_centroid, exposure,
                                         uvw, weight, sigma]))

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
    return da.Array(graph, name, avg.chunks, dtype=dtype)


_row_chan_avg_dims = ("row", "chan", "corr")


def _dask_row_chan_average(row_meta, chan_meta, flag_row=None, weight=None,
                           vis=None, flag=None,
                           weight_spectrum=None, sigma_spectrum=None,
                           chan_bin_size=1):
    """ Average (row,chan,corr)-based dask arrays """

    # We don't know how many rows are in each row chunk,
    # but we can simply divide each channel chunk size by the bin size
    adjust_chunks = {
        "row": lambda r: np.nan,
        "chan": lambda c: (c + chan_bin_size - 1) // chan_bin_size
    }

    flag_row_dims = None if flag_row is None else ("row",)
    weight_dims = None if weight is None else ("row",)
    vis_dims = None if vis is None else _row_chan_avg_dims
    flag_dims = None if flag is None else _row_chan_avg_dims
    ws_dims = None if weight_spectrum is None else _row_chan_avg_dims
    ss_dims = None if sigma_spectrum is None else _row_chan_avg_dims

    avg = da.blockwise(row_chan_average, _row_chan_avg_dims,
                       row_meta, ("row",),
                       chan_meta, ("chan",),
                       flag_row, flag_row_dims,
                       weight, weight_dims,
                       vis, vis_dims,
                       flag, flag_dims,
                       weight_spectrum, ws_dims,
                       sigma_spectrum, ss_dims,
                       adjust_chunks=adjust_chunks,
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
    return da.Array(graph, name, avg.chunks, dtype=dtype)


def _dask_chan_average(chan_meta, chan_freq=None, chan_width=None,
                       chan_bin_size=1):
    adjust_chunks = {
        "chan": lambda c: (c + chan_bin_size - 1) // chan_bin_size
    }

    avg = da.blockwise(chan_average, ("chan",),
                       chan_meta, ("chan",),
                       chan_freq, None if chan_freq is None else ("chan",),
                       chan_width, None if chan_width is None else ("chan",),
                       adjust_chunks=adjust_chunks,
                       dtype=np.object)

    tuple_gets = (None if a is None else _getitem_chan(avg, i, a.dtype)
                  for i, a in enumerate([chan_freq, chan_width]))

    return ChannelAverageOutput(*tuple_gets)


def _dask_merge_flags(flag_row, flag):
    """ Perform flag merging on dask arrays """
    if flag_row is None and flag is not None:
        return da.blockwise(merge_flags, "r",
                            flag_row, None,
                            flag, "rfc",
                            concatenate=True,
                            dtype=flag.dtype)
    elif flag_row is not None and flag is None:
        return da.blockwise(merge_flags, "r",
                            flag_row, "r",
                            None, None,
                            dtype=flag_row.dtype)
    elif flag_row is not None and flag is not None:
        return da.blockwise(merge_flags, "r",
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
                     vis=None, flag=None,
                     weight_spectrum=None, sigma_spectrum=None,
                     time_bin_secs=1.0, chan_bin_size=1):

    row_chan_arrays = (vis, flag, weight_spectrum, sigma_spectrum)

    # The flow of this function should match that of the numba
    # time_and_channel implementation

    # Merge flag_row and flag arrays
    flag_row = _dask_merge_flags(flag_row, flag)

    # Generate row mapping metadata
    row_meta = _dask_row_mapper(time, interval,
                                antenna1, antenna2,
                                flag_row=flag_row,
                                time_bin_secs=time_bin_secs)

    # Generate channel mapping metadata
    chan_meta = _row_chan_metadata(row_chan_arrays, chan_bin_size)

    # Average row data
    row_data = _dask_row_average(row_meta, antenna1, antenna2,
                                 flag_row=flag_row,
                                 time_centroid=time_centroid,
                                 exposure=exposure, uvw=uvw,
                                 weight=weight, sigma=sigma)

    # Average channel data
    row_chan_data = _dask_row_chan_average(row_meta, chan_meta,
                                           flag_row=flag_row, weight=weight,
                                           vis=vis, flag=flag,
                                           weight_spectrum=weight_spectrum,
                                           sigma_spectrum=sigma_spectrum,
                                           chan_bin_size=chan_bin_size)

    chan_data = _dask_chan_average(chan_meta, chan_freq=chan_freq,
                                   chan_width=chan_width)

    # Merge output tuples
    return AverageOutput(_getitem_row(row_meta, 1, time.dtype),
                         _getitem_row(row_meta, 2, interval.dtype),
                         (_getitem_row(row_meta, 3, flag_row.dtype)
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
                         row_chan_data.vis,
                         row_chan_data.flag,
                         row_chan_data.weight_spectrum,
                         row_chan_data.sigma_spectrum)


try:
    time_and_channel.__doc__ = AVERAGING_DOCS.substitute(
                                    array_type=":class:`dask.array.Array`")
except AttributeError:
    pass
