# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from africanus.averaging.time_and_channel_avg import (
                        time_and_channel as np_time_and_channel,
                        TIME_AND_CHANNEL_DOCS)
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


@requires_optional("dask.array", dask_import_error)
def time_and_channel(time, ant1, ant2, vis, flags,
                     avg_time=None, avg_chan=None,
                     return_time=False,
                     return_antenna=False):

    # We're not really sure how many rows we'll end up with in each chunk
    row_chunks = tuple(np.nan for c in vis.chunks[0])
    # Channel averaging is more predictable
    chan_chunks = tuple((c + avg_chan - 1) // avg_chan for c in vis.chunks[1])

    corr_dims = tuple("corr-%d" % i for i in range(len(vis.shape[2:])))
    vis_dims = ("row", "chan") + corr_dims

    token = tokenize(time, ant1, ant2, vis, flags, avg_time, avg_chan)
    tc_name = "time-and-channel-" + token

    layers = db.blockwise(np_time_and_channel, tc_name, vis_dims,
                          time.name, ("row",),
                          ant1.name, ("row",),
                          ant2.name, ("row",),
                          vis.name, vis_dims,
                          flags.name, vis_dims,
                          avg_time=avg_time,
                          avg_chan=avg_chan,
                          return_time=return_time,
                          return_antenna=return_antenna,
                          numblocks={
                            time.name: time.numblocks,
                            ant1.name: ant1.numblocks,
                            ant2.name: ant2.numblocks,
                            vis.name: vis.numblocks,
                            flags.name: flags.numblocks,
                          })

    deps = (time, ant1, ant2, vis, flags)
    graph = HighLevelGraph.from_collections(tc_name, layers, deps)

    # The numpy function we're wrapping may return a tuple
    # depending on whether we're asking it to return
    # averaged times and antennas. In this cases we need to
    # create dask arrays that encapsulate operations which
    # called getitem on these tuples.

    if not return_time and not return_antenna:
        vis_chunks = (row_chunks, chan_chunks) + vis.chunks[2:]
        return da.Array(graph, tc_name, vis_chunks, dtype=vis.dtype)
    elif return_time and not return_antenna:
        # Create an array extracting visibilities out of the tuple
        name0 = "time-and-channel-getitem0-" + tokenize(token, 0)
        layers0 = db.blockwise(_getitem_tup, name0, vis_dims,
                               tc_name, vis_dims,
                               0, None,
                               0, None,
                               numblocks={
                                tc_name: vis.numblocks
                               })
        graph0 = HighLevelGraph.from_collections(name0, layers0, ())
        graph0 = HighLevelGraph.merge(graph, graph0)
        vis_chunks = (row_chunks, chan_chunks) + vis.chunks[2:]
        vis = da.Array(graph0, name0, vis_chunks, dtype=vis.dtype)

        # The averaged times, ant1 and ant2 are computed multiple times
        # if there are multiple channel or correlation blocks.
        # This is wasted computation and we simply take the
        # time/ant1/ant2 from that of the first channel/correlation blocks
        nextra_blocks = len(vis.numblocks[1:])
        extra_blocks = (1,)*nextra_blocks
        numblocks = {tc_name: (vis.numblocks[0],) + extra_blocks}

        name1 = "time-and-channel-getitem1-" + tokenize(token, 1)
        layers1 = db.blockwise(_getitem_tup, name1, ("row",),
                               tc_name, vis_dims,
                               nextra_blocks, None,
                               1, None,
                               numblocks=numblocks)
        graph1 = HighLevelGraph.from_collections(name1, layers1, ())
        graph1 = HighLevelGraph.merge(graph, graph1)
        time = da.Array(graph1, name1, (vis.chunks[0],), dtype=time.dtype)

        return vis, time
    elif not return_time and return_antenna:
        # Create an array extracting visibilities out of the tuple
        name0 = "time-and-channel-getitem0-" + tokenize(token, 0)
        layers0 = db.blockwise(_getitem_tup, name0, vis_dims,
                               tc_name, vis_dims,
                               0, None,
                               0, None,
                               numblocks={
                                tc_name: vis.numblocks
                               })
        graph0 = HighLevelGraph.from_collections(name0, layers0, ())
        graph0 = HighLevelGraph.merge(graph, graph0)
        vis_chunks = (row_chunks, chan_chunks) + vis.chunks[2:]
        vis = da.Array(graph0, name0, vis_chunks, dtype=vis.dtype)

        name1 = "time-and-channel-getitem1-" + tokenize(token, 1)
        layers1 = db.blockwise(_getitem_tup, name1, ("row",),
                               tc_name, vis_dims,
                               nextra_blocks, None,
                               1, None,
                               numblocks=numblocks)
        graph1 = HighLevelGraph.from_collections(name1, layers1, ())
        graph1 = HighLevelGraph.merge(graph, graph1)
        ant1 = da.Array(graph1, name1, (vis.chunks[0],), dtype=ant1.dtype)

        name2 = "time-and-channel-getitem2-" + tokenize(token, 2)
        layers2 = db.blockwise(_getitem_tup, name2, ("row",),
                               tc_name, vis_dims,
                               nextra_blocks, None,
                               2, None,
                               numblocks=numblocks)
        graph2 = HighLevelGraph.from_collections(name1, layers2, ())
        graph2 = HighLevelGraph.merge(graph, graph2)
        ant2 = da.Array(graph2, name2, (vis.chunks[0],), dtype=ant1.dtype)

        return vis, ant1, ant2

    elif return_time and return_antenna:
        # Create an array extracting visibilities out of the tuple
        name0 = "time-and-channel-getitem0-" + tokenize(token, 0)
        layers0 = db.blockwise(_getitem_tup, name0, vis_dims,
                               tc_name, vis_dims,
                               0, None,
                               0, None,
                               numblocks={
                                tc_name: vis.numblocks
                               })
        graph0 = HighLevelGraph.from_collections(name0, layers0, ())
        graph0 = HighLevelGraph.merge(graph, graph0)
        vis_chunks = (row_chunks, chan_chunks) + vis.chunks[2:]
        vis = da.Array(graph0, name0, vis_chunks, dtype=vis.dtype)

        nextra_blocks = len(vis.numblocks[1:])
        extra_blocks = (1,)*nextra_blocks
        numblocks = {tc_name: (vis.numblocks[0],) + extra_blocks}

        name1 = "time-and-channel-getitem1-" + tokenize(token, 1)
        layers1 = db.blockwise(_getitem_tup, name1, ("row",),
                               tc_name, vis_dims,
                               nextra_blocks, None,
                               1, None,
                               numblocks=numblocks)
        graph1 = HighLevelGraph.from_collections(name1, layers1, ())
        graph1 = HighLevelGraph.merge(graph, graph1)
        time = da.Array(graph1, name1, (vis.chunks[0],), dtype=time.dtype)

        name2 = "time-and-channel-getitem2-" + tokenize(token, 2)
        layers2 = db.blockwise(_getitem_tup, name2, ("row",),
                               tc_name, vis_dims,
                               nextra_blocks, None,
                               2, None,
                               numblocks=numblocks)
        graph2 = HighLevelGraph.from_collections(name1, layers2, ())
        graph2 = HighLevelGraph.merge(graph, graph2)
        ant1 = da.Array(graph2, name2, (vis.chunks[0],), dtype=ant1.dtype)

        name3 = "time-and-channel-getitem3-" + tokenize(token, 3)
        layers3 = db.blockwise(_getitem_tup, name3, ("row",),
                               tc_name, vis_dims,
                               nextra_blocks, None,
                               3, None,
                               numblocks=numblocks)
        graph3 = HighLevelGraph.from_collections(name1, layers3, ())
        graph3 = HighLevelGraph.merge(graph, graph3)
        ant2 = da.Array(graph3, name3, (vis.chunks[0],), dtype=ant2.dtype)

        return vis, time, ant1, ant2


try:
    time_and_channel.__doc__ = TIME_AND_CHANNEL_DOCS.substitute(
                                    array_type=":class:`dask.array.Array`")
except AttributeError:
    pass
