# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from itertools import product
from functools import wraps
from operator import mul

import numpy as np

from africanus.compatibility import range, reduce, Mapping
from africanus.util.requirements import requires_optional

from africanus.rime.predict import (PREDICT_DOCS, predict_checks,
                                    predict_vis as np_predict_vis)

try:
    from dask.blockwise import blockwise
    from dask.base import tokenize
    import dask.array as da
    from dask.highlevelgraph import HighLevelGraph
except ImportError as e:
    opt_import_error = e
else:
    opt_import_error = None


def _source_stream_blocks(source_blocks, streams):
    return (source_blocks + streams - 1) // streams


def _extract_blocks(time_index, dde1_jones, source_coh, dde2_jones):
    """
    Returns
    -------
    blocks : tuple
        :code:`(source, row, ant, chan, corr1, ..., corrn)
    """

    if dde1_jones is not None:
        return ((dde1_jones.numblocks[0], time_index.numblocks[0]) +
                (1, dde1_jones.numblocks[3]) +
                dde1_jones.numblocks[4:])
    elif source_coh is not None:
        return (source_coh.numblocks[:2] +
                (1, source_coh.numblocks[2]) +
                source_coh.numblocks[3:])
    else:
        raise ValueError("need ddes or coherencies")


def _extract_chunks(time_index, dde1_jones, source_coh, dde2_jones):
    """
    Returns
    -------
    chunks : tuple
        :code:`(source, row, chan, corr1, ..., corrn)
    """

    if dde1_jones is not None:
        return ((dde1_jones.chunks[0], time_index.chunks[0]) +
                (dde1_jones.chunks[3],) +
                dde1_jones.chunks[4:])
    elif source_coh is not None:
        return source_coh.chunks
    else:
        raise ValueError("need ddes or coherencies")


class CoherencyStreamReduction(Mapping):
    """
    tl;dr this is a dictionary that is expanded in place when
    first acccessed. Saves memory when pickled for sending
    to the dask scheduler.

    See :class:`dask.blockwise.Blockwise` for further insight.

    Produces graph serially summing coherencies in
    ``stream`` parallel streams.
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

        self.blocks = _extract_blocks(time_index, dde1_jones,
                                      source_coh, dde2_jones)
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
        (source, row, _, chan), corr = self.blocks[:4], self.blocks[4:]
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
         chan_blocks), corr_blocks = self.blocks[:4], self.blocks[4:]

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


class CoherencyFinalReduction(Mapping):
    """
    tl;dr this is a dictionary that is expanded in place when
    first acccessed. Saves memory when pickled for sending
    to the dask scheduler.

    See :class:`dask.blockwise.Blockwise` for further insight.

    Produces graph reducing results of ``stream`` parallel streams in
    CoherencyStreamReduction.
    """

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
        (source, row, _, chan), corrs = self.blocks[:4], self.blocks[4:]
        return reduce(mul, (source, row, chan) + corrs, 1)

    def _create_dict(self):
        (source, row, _, chan), corrs = self.blocks[:4], self.blocks[4:]

        # Iterator of block id's for row, channel and correlation blocks
        # We don't reduce over these dimensions
        block_ids = enumerate(product(range(row), range(chan),
                                      *[range(cb) for cb in corrs]))

        source_block_chunks = _source_stream_blocks(source, self.streams)

        layers = {}

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


@wraps(np_predict_vis)
def _predict_coh_wrapper(time_index, antenna1, antenna2,
                         dde1_jones, source_coh, dde2_jones,
                         die1_jones, base_vis, die2_jones):

    return (np_predict_vis(time_index, antenna1, antenna2,
                           # dde1_jones loses the 'ant' dim
                           dde1_jones[0] if dde1_jones else None,
                           # source_coh loses the 'source' dim
                           source_coh,
                           # dde2_jones loses the 'source' and 'ant' dims
                           dde2_jones[0] if dde2_jones else None,
                           # die1_jones loses the 'ant' dim
                           die1_jones[0] if die1_jones else None,
                           base_vis,
                           # die2_jones loses the 'ant' dim
                           die2_jones[0] if die2_jones else None)
            # Introduce an extra dimension (source dim reduced to 1)
            [None, ...])


@wraps(np_predict_vis)
def _predict_dies_wrapper(time_index, antenna1, antenna2,
                          dde1_jones, source_coh, dde2_jones,
                          die1_jones, base_vis, die2_jones):

    return np_predict_vis(time_index, antenna1, antenna2,
                          # dde1_jones loses the 'source' and 'ant' dims
                          dde1_jones[0][0] if dde1_jones else None,
                          # source_coh loses the 'source' dim
                          source_coh[0] if source_coh else None,
                          # dde2_jones loses the 'source' and 'ant' dims
                          dde2_jones[0][0] if dde2_jones else None,
                          # die1_jones loses the 'ant' dim
                          die1_jones[0] if die1_jones else None,
                          base_vis,
                          # die2_jones loses the 'ant' dim
                          die2_jones[0] if die2_jones else None)


def stream_reduction(time_index, antenna1, antenna2,
                     dde1_jones, source_coh, dde2_jones,
                     predict_check_tup, out_dtype, streams):
    """
    Reduces source coherencies + ddes over the source dimension in
    ``N`` parallel streams.

    This is accomplished by calling predict_vis on on ddes and source
    coherencies to produce visibilities which are passed into
    the `base_vis` argument of ``predict_vis`` for the next chunk.
    """

    # Unique name and token for this operation
    token = tokenize(time_index, antenna1, antenna2,
                     dde1_jones, source_coh, dde2_jones,
                     streams)

    name = 'stream-coherency-reduction-' + token

    # Number of dim blocks
    blocks = _extract_blocks(time_index, dde1_jones, source_coh, dde2_jones)
    (src_blocks, row_blocks, _,
     chan_blocks), corr_blocks = blocks[:4], blocks[4:]

    # Total number of other dimension blocks
    nblocks = reduce(mul, (row_blocks, chan_blocks) + corr_blocks, 1)

    # Create the compressed mapping
    layers = CoherencyStreamReduction(time_index, antenna1, antenna2,
                                      dde1_jones, source_coh, dde2_jones,
                                      name, streams)

    # Create the graph
    extra_deps = [a for a in (dde1_jones, source_coh, dde2_jones)
                  if a is not None]
    deps = [time_index, antenna1, antenna2] + extra_deps

    graph = HighLevelGraph.from_collections(name, layers, deps)

    chunks = ((1,) * src_blocks, (1,)*nblocks)
    # This should never be directly computed, reported chunks
    # and dtype don't match the actual data. We create it
    # because it makes chaining HighLevelGraphs easier
    stream_reduction = da.Array(graph, name, chunks, dtype=np.int8)

    name = "coherency-reduction-" + tokenize(stream_reduction)
    layers = CoherencyFinalReduction(name, layers)
    graph = HighLevelGraph.from_collections(name, layers, [stream_reduction])

    chunks = _extract_chunks(time_index, dde1_jones, source_coh, dde2_jones)
    return da.Array(graph, name, chunks[1:], dtype=out_dtype)


def fan_reduction(time_index, antenna1, antenna2,
                  dde1_jones, source_coh, dde2_jones,
                  predict_check_tup, out_dtype):
    """ Does a standard dask tree reduction over source coherencies """
    (have_ddes1, have_coh, have_ddes2,
     have_dies1, have_bvis, have_dies2) = predict_check_tup

    have_ddes = have_ddes1 and have_ddes2

    if have_ddes:
        cdims = tuple("corr-%d" % i for i in range(len(dde1_jones.shape[4:])))
    elif have_coh:
        cdims = tuple("corr-%d" % i for i in range(len(source_coh.shape[3:])))
    else:
        raise ValueError("need ddes or source coherencies")

    ajones_dims = ("src", "row", "ant", "chan") + cdims

    # Setup
    # 1. Optional blockwise arguments
    # 2. Optional numblocks kwarg
    # 3. HighLevelGraph dependencies
    bw_args = [time_index.name, ("row",),
               antenna1.name, ("row",),
               antenna2.name, ("row",)]
    numblocks = {
        time_index.name: time_index.numblocks,
        antenna1.name: antenna1.numblocks,
        antenna2.name: antenna2.numblocks
    }

    # Dependencies
    deps = [time_index, antenna1, antenna2]

    # Handle presence/absence of dde1_jones
    if have_ddes:
        bw_args.extend([dde1_jones.name, ajones_dims])
        numblocks[dde1_jones.name] = dde1_jones.numblocks
        deps.append(dde1_jones)
        other_chunks = dde1_jones.chunks[3:]
        src_chunks = dde1_jones.chunks[0]
    else:
        bw_args.extend([None, None])

    # Handle presence/absence of source_coh
    if have_coh:
        bw_args.extend([source_coh.name, ("src", "row", "chan") + cdims])
        numblocks[source_coh.name] = source_coh.numblocks
        deps.append(source_coh)
        other_chunks = source_coh.chunks[2:]
        src_chunks = source_coh.chunks[0]
    else:
        bw_args.extend([None, None])

    # Handle presence/absence of dde2_jones
    if have_ddes:
        bw_args.extend([dde2_jones.name, ajones_dims])
        numblocks[dde2_jones.name] = dde2_jones.numblocks
        deps.append(dde2_jones)
        other_chunks = dde2_jones.chunks[3:]
        src_chunks = dde2_jones.chunks[0]
    else:
        bw_args.extend([None, None])

    # die1_jones, base_vis and die2_jones absent for this part of the graph
    bw_args.extend([None, None, None, None, None, None])

    assert len(bw_args) // 2 == 9, len(bw_args) // 2

    token = da.core.tokenize(time_index, antenna1, antenna2,
                             dde1_jones, source_coh, dde2_jones)
    name = "-".join(("predict-vis-sum-coh", token))
    layer = blockwise(_predict_coh_wrapper,
                      name, ("src", "row", "chan") + cdims,
                      *bw_args, numblocks=numblocks)

    graph = HighLevelGraph.from_collections(name, layer, deps)

    # We can infer output chunk sizes from source_coh
    chunks = ((1,)*len(src_chunks), time_index.chunks[0],) + other_chunks

    # Create array
    sum_coherencies = da.Array(graph, name, chunks, dtype=out_dtype)

    # Reduce source axis
    return sum_coherencies.sum(axis=0)


def apply_dies(time_index, antenna1, antenna2,
               die1_jones, base_vis, die2_jones,
               predict_check_tup, out_dtype):
    """ Apply any Direction-Independent Effects and Base Visibilities """

    # Now apply any Direction Independent Effect Terms
    (have_ddes1, have_coh, have_ddes2,
     have_dies1, have_bvis, have_dies2) = predict_check_tup

    have_dies = have_dies1 and have_dies2

    # Generate strings for the correlation dimensions
    # This also has the effect of checking that we have all valid inputs
    if have_dies:
        cdims = tuple("corr-%d" % i for i in range(len(die1_jones.shape[3:])))
    elif have_bvis:
        cdims = tuple("corr-%d" % i for i in range(len(base_vis.shape[2:])))
    else:
        raise ValueError("Missing both antenna and baseline jones terms")

    # In the case of predict_vis, the "row" and "time" dimensions
    # are intimately related -- a contiguous series of rows
    # are related to a contiguous series of timesteps.
    # This means that the number of chunks of these
    # two dimensions must match even though the chunk sizes may not.
    # blockwise insists on matching chunk sizes.
    # For this reason, we use the lower level blockwise and
    # substitute "row" for "time" in arrays such as dde1_jones
    # and die1_jones.
    gjones_dims = ("row", "ant", "chan") + cdims

    # Setup
    # 1. Optional blockwise arguments
    # 2. Optional numblocks kwarg
    # 3. HighLevelGraph dependencies
    bw_args = [time_index.name, ("row",),
               antenna1.name, ("row",),
               antenna2.name, ("row",)]
    numblocks = {
        time_index.name: time_index.numblocks,
        antenna1.name: antenna1.numblocks,
        antenna2.name: antenna2.numblocks
    }

    deps = [time_index, antenna1, antenna2]

    # dde1_jones, source_coh and dde2_jones not present
    # these are already applied into sum_coherencies
    bw_args.extend([None, None, None, None, None, None])

    if have_dies:
        bw_args.extend([die1_jones.name, gjones_dims])
        numblocks[die1_jones.name] = die1_jones.numblocks
        deps.append(die1_jones)
        other_chunks = die1_jones.chunks[2:]
    else:
        bw_args.extend([None, None])

    if have_bvis:
        bw_args.extend([base_vis.name, ("row", "chan") + cdims])
        numblocks[base_vis.name] = base_vis.numblocks
        deps.append(base_vis)
        other_chunks = base_vis.chunks[1:]
    else:
        bw_args.extend([None, None])

    if have_dies:
        bw_args.extend([die2_jones.name, gjones_dims])
        numblocks[die2_jones.name] = die2_jones.numblocks
        deps.append(die2_jones)
        other_chunks = die2_jones.chunks[2:]
    else:
        bw_args.extend([None, None])

    assert len(bw_args) // 2 == 9

    token = da.core.tokenize(time_index, antenna1, antenna2,
                             die1_jones, base_vis, die2_jones)
    name = '-'.join(("predict-vis-apply-dies", token))
    layer = blockwise(_predict_dies_wrapper,
                      name, ("row", "chan") + cdims,
                      *bw_args, numblocks=numblocks)

    graph = HighLevelGraph.from_collections(name, layer, deps)
    chunks = (time_index.chunks[0],) + other_chunks

    return da.Array(graph, name, chunks, dtype=out_dtype)


@requires_optional('dask.array', opt_import_error)
def predict_vis(time_index, antenna1, antenna2,
                dde1_jones=None, source_coh=None, dde2_jones=None,
                die1_jones=None, base_vis=None, die2_jones=None,
                streams=None):

    predict_check_tup = predict_checks(time_index, antenna1, antenna2,
                                       dde1_jones, source_coh, dde2_jones,
                                       die1_jones, base_vis, die2_jones)

    (have_ddes1, have_coh, have_ddes2,
     have_dies1, have_bvis, have_dies2) = predict_check_tup

    have_ddes = have_ddes1 and have_ddes2

    if have_ddes:
        if dde1_jones.shape[2] != dde1_jones.chunks[2][0]:
            raise ValueError("Subdivision of antenna dimension into "
                             "multiple chunks is not supported.")

        if dde2_jones.shape[2] != dde2_jones.chunks[2][0]:
            raise ValueError("Subdivision of antenna dimension into "
                             "multiple chunks is not supported.")

        if dde1_jones.chunks != dde2_jones.chunks:
            raise ValueError("dde1_jones.chunks != dde2_jones.chunks")

        if len(dde1_jones.chunks[1]) != len(time_index.chunks[0]):
            raise ValueError("Number of row chunks (%s) does not equal "
                             "number of time chunks (%s)." %
                             (time_index.chunks[0], dde1_jones.chunks[1]))

    have_dies = have_dies1 and have_dies2

    if have_dies:
        if die1_jones.shape[1] != die1_jones.chunks[1][0]:
            raise ValueError("Subdivision of antenna dimension into "
                             "multiple chunks is not supported.")

        if die2_jones.shape[1] != die2_jones.chunks[1][0]:
            raise ValueError("Subdivision of antenna dimension into "
                             "multiple chunks is not supported.")

        if die1_jones.chunks != die2_jones.chunks:
            raise ValueError("die1_jones.chunks != die2_jones.chunks")

        if len(die1_jones.chunks[0]) != len(time_index.chunks[0]):
            raise ValueError("Number of row chunks (%s) does not equal "
                             "number of time chunks (%s)." %
                             (time_index.chunks[0], die1_jones.chunks[1]))

    # Infer the output dtype
    dtype_arrays = [dde1_jones, source_coh, dde2_jones, die1_jones, die2_jones]
    out_dtype = np.result_type(*(np.dtype(a.dtype.name)
                                 for a in dtype_arrays if a is not None))

    # Apply direction dependent effects
    if have_coh or have_ddes:
        # We create separate graphs for computing coherencies and applying
        # the gains because coherencies are chunked over source which
        # must be summed and added to the (possibly present) base visibilities
        if streams is not None:
            sum_coherencies = stream_reduction(time_index, antenna1, antenna2,
                                               dde1_jones, source_coh,
                                               dde2_jones, predict_check_tup,
                                               out_dtype, streams=streams)
        else:
            sum_coherencies = fan_reduction(time_index, antenna1, antenna2,
                                            dde1_jones, source_coh, dde2_jones,
                                            predict_check_tup, out_dtype)
    else:
        assert have_dies or have_bvis
        sum_coherencies = None

    # No more effects to apply, return at this point
    if not have_dies and not have_bvis:
        return sum_coherencies

    # Add coherencies to the base visibilities
    if sum_coherencies is not None:
        if not have_bvis:
            # Set base_vis = summed coherencies
            base_vis = sum_coherencies
            predict_check_tup = (have_ddes1, have_coh, have_ddes2,
                                 have_dies1, True, have_dies2)
        else:
            base_vis += sum_coherencies

    # Apply direction independent effects
    return apply_dies(time_index, antenna1, antenna2,
                      die1_jones, base_vis, die2_jones,
                      predict_check_tup, out_dtype)


EXTRA_DASK_ARGS = """
streams : int, optional
    Specifies the degree of parallelism along the source dimension.
    By default, dask uses a tree style reduction algorithm which can
    require large amounts of memory. Specifying this parameter
    constrains the dask graph to serially sum coherencies in a
    specified number of streams, reducing overall memory usage.

    If ``None``, defaults to a standard, memory-intensive tree style
    algorithm.

    Defaults to 1, which means that the source coherencies for each
    visibility chunk are serially summed, meaning that parallelism
    will only exists along the row and chan dimensions.
"""

EXTRA_DASK_NOTES = """
* The ``ant`` dimension should only contain a single chunk equal
  to the number of antenna. Since each ``row`` can contain
  any antenna, random access must be preserved along this dimension.
* The chunks in the ``row`` and ``time`` dimension **must** align.
  This subtle point **must be understood otherwise
  invalid results will be produced** by the chunking scheme.
  In the example below
  we have four unique time indices :code:`[0,1,2,3]`, and
  four unique antenna :code:`[0,1,2,3]` indexing :code:`10` rows.

  .. code-block:: python

      #  Row indices into the time/antenna indexed arrays
      time_idx = np.asarray([0,0,1,1,2,2,2,2,3,3])
      ant1 = np.asarray(    [0,0,0,0,1,1,1,2,2,3]
      ant2 = np.asarray(    [0,1,2,3,1,2,3,2,3,3])


  A reasonable chunking scheme for the
  ``row`` and ``time`` dimension would be :code:`(4,4,2)`
  and :code:`(2,1,1)` respectively.
  Another way of explaining this is that the first
  four rows contain two unique timesteps, the second four
  rows contain one unique timestep and the last two rows
  contain one unique timestep.

  Some rules of thumb:

  1. The number chunks in ``row`` and ``time`` must match
     although the individual chunk sizes need not.
  2. Unique timesteps should not be split across row chunks.
  3. For a Measurement Set whose rows are ordered on the
     ``TIME`` column, the following is a good way of obtaining
     the row chunking strategy:

     .. code-block:: python

        import numpy as np
        import pyrap.tables as pt

        ms = pt.table("data.ms")
        times = ms.getcol("TIME")
        unique_times, chunks = np.unique(times, return_counts=True)

  4. Use :func:`~africanus.util.shapes.aggregate_chunks`
     to aggregate multiple ``row`` and ``time``
     chunks into chunks large enough such that functions operating
     on the resulting data can drop the GIL and spend time
     processing the data. Expanding the previous example:

     .. code-block:: python

        # Aggregate row
        utimes = unique_times.size
        # Single chunk for each unique time
        time_chunks = (1,)*utimes
        # Aggregate row chunks into chunks <= 10000
        aggregate_chunks((chunks, time_chunks), (10000, utimes))
"""

try:
    predict_vis.__doc__ = PREDICT_DOCS.substitute(
                                array_type=":class:`dask.array.Array`",
                                get_time_index=":code:`time.map_blocks("
                                               "lambda a: np.unique(a, "
                                               "return_inverse=True)[1])`",
                                extra_args=EXTRA_DASK_ARGS,
                                extra_notes=EXTRA_DASK_NOTES)
except AttributeError:
    pass
