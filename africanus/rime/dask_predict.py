# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from itertools import product
from operator import mul

import numpy as np

from africanus.compatibility import range, reduce
from africanus.util.requirements import requires_optional

from africanus.rime.predict import predict_vis as np_predict_vis

try:
    from dask.compatibility import Mapping
    from dask.base import tokenize
    import dask.array as da
    from dask.highlevelgraph import HighLevelGraph
except ImportError as e:
    opt_import_error = e
else:
    opt_import_error = None


def _source_stream_blocks(source_blocks, streams):
    return (source_blocks + streams - 1) // streams


def _extract_block_dims(time_index, antenna1, antenna2,
                        dde1_jones, source_coh, dde2_jones):
    # Number of dim blocks
    src_blocks = dde1_jones.numblocks[0]
    row_blocks = source_coh.numblocks[1]
    ant_blocks = 1
    chan_blocks = source_coh.numblocks[2]
    corr_blocks = source_coh.numblocks[3:]

    return src_blocks, row_blocks, ant_blocks, chan_blocks, corr_blocks


@requires_optional("dask.array", opt_import_error)
class CoherencyStreamReduction(Mapping):
    """
    tl;dr this is a dictionary that is expanded in place when
    first acccessed. Saves memory when pickled for sending
    to the dask scheduler.

    See :class:`dask.blockwise.Blockwise` for further insight.
    """
    def __init__(self, time_index, antenna1, antenna2,
                 dde1_jones, source_coh, dde2_jones,
                 out_name, streams):
        self.time_index_name = None if time_index is None else time_index.name
        self.ant1_name = None if antenna1 is None else antenna1.name
        self.ant2_name = None if antenna2 is None else antenna2.name
        self.dde1_name = None if dde1_jones is None else dde1_jones.name
        self.coh_name = None if source_coh is None else source_coh.name
        self.dde2_name = None if dde2_jones is None else dde2_jones.name

        self.out_name = out_name

        self.blocks = _extract_block_dims(time_index, antenna1, antenna2,
                                          dde1_jones, source_coh, dde2_jones)
        self.streams = streams

    @property
    def _dict(self):
        if hasattr(self, "_cached_dict"):
            return self._cached_dict
        else:
            self._cached_dict = self._create_dict()
            return self._cached_dict

    def __getitem__(self, key):
        return self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        # Extract dimension blocks
        (source, row, ant, chan, corr) = self.blocks
        return reduce(mul, (source, row, chan) + corr, 1)

    def _create_dict(self):
        # Graph dictionary
        layers = {}

        # For loop performance
        out_name = self.out_name
        ti = self.time_index_name
        a1 = self.ant1_name
        a2 = self.ant2_name
        dde1 = self.dde1_name
        coh = self.coh_name
        dde2 = self.dde2_name

        # Extract dimension blocks
        (source_blocks, row_blocks, ant_blocks,
         chan_blocks, corr_blocks) = self.blocks

        assert ant_blocks == 1
        ab = 0

        # Subdivide number of source blocks by number of streams
        source_block_chunks = _source_stream_blocks(source_blocks,
                                                    self.streams)

        # Iterator of block id's for row, channel and correlation blocks
        # We don't reduce over these dimensions
        block_ids = enumerate(product(range(row_blocks), range(chan_blocks),
                                      *[range(cb) for cb in corr_blocks]))

        for flat_bid, bid in block_ids:
            rb, fb = bid[0:2]
            cb = bid[2:]

            # Create the streamed reduction proper.
            # For a stream, the base visibilities are set to the result
            # of the previous result in the stream (last_key)
            for sb_start in range(0, source_blocks, source_block_chunks):
                sb_end = min(sb_start + source_block_chunks, source_blocks)
                last_key = None

                for sb in range(sb_start, sb_end):
                    # Dask task object calling predict vis
                    task = (np_predict_vis,
                            (ti, rb), (a1, rb), (a2, rb),
                            (dde1, sb, rb, ab, fb) + cb if dde1 else None,
                            (coh, sb, rb, fb) + cb if coh else None,
                            (dde2, sb, rb, ab, fb) + cb if dde2 else None,
                            None, last_key, None)

                    key = (out_name, sb, flat_bid)
                    layers[key] = task
                    last_key = key

        return layers


@requires_optional('dask.array', opt_import_error)
class CoherencyFinalReduction(Mapping):
    def __init__(self, out_name, coherency_stream_reduction):
        self.in_name = coherency_stream_reduction.out_name
        self.blocks = coherency_stream_reduction.blocks
        self.streams = coherency_stream_reduction.streams
        self.out_name = out_name

    @property
    def _dict(self):
        if hasattr(self, "_cached_dict"):
            return self._cached_dict
        else:
            self._cached_dict = self._create_dict()
            return self._cached_dict

    def __getitem__(self, key):
        return self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        (source, row, ant, chan, corr) = self.blocks
        return reduce(mul, (source, row, chan) + corr, 1)

    def _create_dict(self):
        (source, row, ant, chan, corr) = self.blocks

        # Iterator of block id's for row, channel and correlation blocks
        # We don't reduce over these dimensions
        block_ids = enumerate(product(range(row), range(chan),
                                      *[range(cb) for cb in corr]))

        source_block_chunks = _source_stream_blocks(source, self.streams)

        layers = {}
        last_block_keys = []

        # This looping structure should match
        for flat_bid, bid in block_ids:
            rb, fb = bid[0:2]
            cb = bid[2:]

            last_stream_keys = []

            for sb_start in range(0, source, source_block_chunks):
                sb_end = min(sb_start + source_block_chunks, source)
                key = (sb_end - 1, flat_bid)
                last_stream_keys.append((self.in_name, sb_end - 1, flat_bid))

            key = (self.out_name, rb, fb) + cb
            task = (sum, last_stream_keys)
            layers[key] = task

        return layers


def coherency_reduction(time_index, antenna1, antenna2,
                        dde1_jones, source_coh, dde2_jones,
                        streams):
    # Unique name and token for this operation
    token = tokenize(time_index, antenna1, antenna2,
                     dde1_jones, source_coh, dde2_jones,
                     streams)

    name = 'stream-coherency-reduction-' + token

    # Number of dim blocks
    (src_blocks, row_blocks, ant_blocks,
     chan_blocks, corr_blocks) = _extract_block_dims(time_index,
                                                     antenna1,
                                                     antenna2,
                                                     dde1_jones,
                                                     source_coh,
                                                     dde2_jones)

    if dde1_jones.numblocks[1] != source_coh.numblocks[1]:
        raise ValueError("time and row blocks must match")

    if dde1_jones.numblocks[2] != 1:
        raise ValueError("Chunking along antenna unsupported")

    # Subdivide number of source blocks by number of streams
    src_block_chunks = (src_blocks + streams - 1) // streams

    # Total number of other dimension blocks
    nblocks = reduce(mul, (row_blocks, chan_blocks) + corr_blocks, 1)

    # Create the compressed mapping
    layers = CoherencyStreamReduction(time_index, antenna1, antenna2,
                                      dde1_jones, source_coh, dde2_jones,
                                      name, streams)

    # Create the graph
    deps = [time_index, antenna1, antenna2,
            dde1_jones, source_coh, dde2_jones]
    graph = HighLevelGraph.from_collections(name, layers, deps)

    chunks = ((1,) * src_blocks, (1,)*nblocks)
    # This should never be directly computed, reported chunks
    # and dtype don't match the actual data. We create it
    # because it makes chaining HighLevelGraphs easier
    stream_reduction = da.Array(graph, name, chunks, dtype=np.int8)

    name = "coherency-reduction-" + tokenize(stream_reduction)
    layers = CoherencyFinalReduction(name, layers)
    graph = HighLevelGraph.from_collections(name, layers, [stream_reduction])

    # TODO(sjperkins)
    # Infer the result type properly
    return da.Array(graph, name, source_coh.chunks[1:], dtype=np.complex128)
