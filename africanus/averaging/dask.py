# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from operator import getitem

from africanus.averaging.time_and_channel_mapping import (row_mapper,
                                                          channel_mapper)
from africanus.averaging.time_and_channel_avg import (row_average,
                                                      row_chan_average,
                                                      merge_flags,
                                                      AVERAGING_DOCS,
                                                      AverageOutput,
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


def _getitem_tup(array, d, n):
    """ Recursively index a sequence d times and return element n"""
    for i in range(d):
        array = array[0]

    return array[n]


def _row_chan_metadata(arrays, chan_bin_size):
    for array in arrays:
        if array is None:
            continue

        chan_chunks = tuple((c + chan_bin_size - 1) // chan_bin_size
                            for c in array.chunks[1])
        corr_chunks = array.chunks[2]
        corr_dims = tuple("corr-%d" % i for i in range(len(array.shape[2:])))

        # Create a dask channel mapping structure
        name = "channel-mapper-" + tokenize(array.chunks[1], chan_bin_size)
        layers = {(name, i): (channel_mapper, c, chan_bin_size)
                  for i, c in enumerate(array.chunks[1])}
        graph = HighLevelGraph.from_collections(name, layers, ())
        chunks = (array.chunks[1],)
        chan_mapper = da.Array(graph, name, chunks, dtype=np.object)

        return chan_chunks, corr_chunks, corr_dims, chan_mapper

    return None, None, None, None


def _dask_row_mapper(time_centroid, exposure, antenna1, antenna2,
                     flag_row=None, time_bin_secs=1.0):
    """ Create a dask row mapping structure """
    return da.blockwise(row_mapper, ("row",),
                        time_centroid, ("row",),
                        exposure, ("row",),
                        antenna1, ("row",),
                        antenna2, ("row",),
                        flag_row, None if flag_row is None else ("row",),
                        time_bin_secs=time_bin_secs,
                        dtype=np.object)


def _getitem_row(avg, idx, dtype):
    name = ("row-average-getitem-%d-" % idx) + tokenize(avg, idx)
    layers = db.blockwise(getitem, name, ("row",),
                          avg.name, ("row",),
                          idx, None,
                          numblocks={avg.name: avg.numblocks})
    graph = HighLevelGraph.from_collections(name, layers, (avg,))
    return da.Array(graph, name, avg.chunks, dtype=dtype)


def _dask_row_average(row_meta, ant1, ant2, flag_row=None,
                      time=None, interval=None, uvw=None,
                      weight=None, sigma=None):

    rd = ("row",)
    rcd = ("row", "corr")

    avg = da.blockwise(row_average, rd,
                       row_meta, rd,
                       ant1, rd,
                       ant2, rd,
                       flag_row, None if flag_row is None else rd,
                       time, None if time is None else rd,
                       interval, None if interval is None else rd,
                       uvw, None if uvw is None else ("row", "3"),
                       weight, None if weight is None else rcd,
                       sigma, None if sigma is None else rcd,
                       adjust_chunks={"row": lambda x: np.nan},
                       dtype=np.object)

    tuple_gets = (None if a is None else _getitem_row(avg, i, a.dtype)
                  for i, a in enumerate([ant1, ant2, time, interval,
                                         uvw, weight, sigma]))

    return RowAverageOutput(*tuple_gets)


def _getitem_row_chan(avg, idx, dtype):
    name = ("row-chan-average-getitem-%d-" % idx) + tokenize(avg, idx)
    dim = ("row", "chan", "corr")

    layers = db.blockwise(getitem, name, dim,
                          avg.name, dim,
                          idx, None,
                          numblocks={avg.name: avg.numblocks})

    graph = HighLevelGraph.from_collections(name, layers, (avg,))
    return da.Array(graph, name, avg.chunks, dtype=dtype)


_row_chan_avg_dims = ("row", "chan", "corr")


def _dask_row_chan_average(row_meta, chan_meta, flag_row=None,
                           vis=None, flag=None,
                           weight_spectrum=None, sigma_spectrum=None,
                           chan_bin_size=1):

    # We don't know how many rows are in each row chunk,
    # but we can simply divide each channel chunk size by the bin size
    adjust_chunks = {
        "row": lambda r: np.nan,
        "chan": lambda c: (c + chan_bin_size - 1) // chan_bin_size
    }

    flag_row_dims = None if flag_row is None else ("row",)
    vis_dims = None if vis is None else _row_chan_avg_dims
    flag_dims = None if flag is None else _row_chan_avg_dims
    ws_dims = None if weight_spectrum is None else _row_chan_avg_dims
    ss_dims = None if sigma_spectrum is None else _row_chan_avg_dims

    avg = da.blockwise(row_chan_average, _row_chan_avg_dims,
                       row_meta, ("row",),
                       chan_meta, ("chan",),
                       flag_row, flag_row_dims,
                       vis, vis_dims,
                       flag, flag_dims,
                       weight_spectrum, ws_dims,
                       sigma_spectrum, ss_dims,
                       chan_bin_size=chan_bin_size,
                       adjust_chunks=adjust_chunks,
                       dtype=np.object)

    tuple_gets = (None if a is None else _getitem_row_chan(avg, i, a.dtype)
                  for i, a in enumerate([vis, flag,
                                         weight_spectrum,
                                         sigma_spectrum]))

    return RowChanAverageOutput(*tuple_gets)


def _merge_flags_wrapper(flag_row, flag):
    return merge_flags(flag_row, flag[0][0])


def _dask_merge_flags(flag_row, flag):
    if flag_row is None and flag is not None:
        return da.blockwise(_merge_flags_wrapper, "r",
                            flag_row, None,
                            flag, "rfc",
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
                            dtype=flag_row.dtype)
    else:
        return None


@requires_optional("dask.array", dask_import_error)
def time_and_channel(time_centroid, exposure, antenna1, antenna2,
                     time=None, interval=None, flag_row=None,
                     uvw=None, weight=None, sigma=None,
                     vis=None, flag=None,
                     weight_spectrum=None, sigma_spectrum=None,
                     time_bin_secs=1.0, chan_bin_size=1):

    row_chan_arrays = (vis, flag, weight_spectrum, sigma_spectrum)

    (chan_chunks, corr_chunks,
     corr_dims, chan_meta) = _row_chan_metadata(row_chan_arrays, chan_bin_size)

    flag_row = _dask_merge_flags(flag_row, flag)

    row_meta = _dask_row_mapper(time_centroid, exposure,
                                antenna1, antenna2,
                                flag_row=flag_row,
                                time_bin_secs=time_bin_secs)

    row_data = _dask_row_average(row_meta, antenna1, antenna2,
                                 flag_row=flag_row, time=time,
                                 interval=interval, uvw=uvw,
                                 weight=weight, sigma=sigma)

    chan_data = _dask_row_chan_average(row_meta, chan_meta, flag_row=flag_row,
                                       vis=vis, flag=flag,
                                       weight_spectrum=weight_spectrum,
                                       sigma_spectrum=sigma_spectrum,
                                       chan_bin_size=chan_bin_size)

    return AverageOutput(_getitem_row(row_meta, 1, time_centroid.dtype),
                         _getitem_row(row_meta, 2, exposure.dtype),
                         (_getitem_row(row_meta, 3, flag_row.dtype)
                          if flag_row is not None else None),
                         row_data.antenna1,
                         row_data.antenna2,
                         row_data.time,
                         row_data.interval,
                         row_data.uvw,
                         row_data.weight,
                         row_data.sigma,
                         chan_data.vis,
                         chan_data.flag,
                         chan_data.weight_spectrum,
                         chan_data.sigma_spectrum)


try:
    time_and_channel.__doc__ = AVERAGING_DOCS.substitute(
                                    array_type=":class:`dask.array.Array`")
except AttributeError:
    pass
