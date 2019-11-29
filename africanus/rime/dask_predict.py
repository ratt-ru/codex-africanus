# -*- coding: utf-8 -*-


try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

from functools import reduce
from itertools import product
from operator import mul

import numpy as np

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


def _predict_coh_wrapper(time_index, antenna1, antenna2,
                         dde1_jones, source_coh, dde2_jones,
                         die1_jones, base_vis, die2_jones,
                         reduce_source=False):

    if reduce_source:
        dde1_jones = dde1_jones[0] if dde1_jones else None
        source_coh = source_coh[0] if isinstance(source_coh, list) else None
        dde2_jones = dde2_jones[0] if dde2_jones else None

    vis = np_predict_vis(time_index, antenna1, antenna2,
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

    if reduce_source:
        return vis

    return vis[None, ...]


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
                     predict_check_tup, out_dtype):

    (have_ddes1, have_coh, have_ddes2,
     have_dies1, have_bvis, have_dies2) = predict_check_tup

    have_ddes = have_ddes1 and have_ddes2

    if have_ddes:
        cdims = tuple("corr-%d" % i for i in range(len(dde1_jones.shape[4:])))
        src_blocks = dde1_jones.numblocks[0]
    elif have_coh:
        cdims = tuple("corr-%d" % i for i in range(len(source_coh.shape[3:])))
        src_blocks = source_coh.numblocks[0]
    else:
        raise ValueError("need ddes or source coherencies")

    ajones_dims = ("src", "row", "ant", "chan") + cdims
    vis_dims = ("row", "chan") + cdims
    src_coh_dims = ("src", "row", "chan") + cdims

    base_vis = da.blockwise(
        _predict_coh_wrapper, vis_dims,
        time_index, ("row",),
        antenna1, ("row",),
        antenna2, ("row",),
        None if dde1_jones is None else dde1_jones.blocks[0, ...],
        None if dde1_jones is None else ajones_dims,

        None if source_coh is None else source_coh.blocks[0, ...],
        None if source_coh is None else src_coh_dims,

        None if dde2_jones is None else dde2_jones.blocks[0, ...],
        None if dde2_jones is None else ajones_dims,

        None, None,
        None, None,
        None, None,
        reduce_source=True,
        # time+row dimension chunks are equivalent but differently sized
        align_arrays=False,
        # Force row dimension to take row chunking scheme,
        # instead of time chunking scheme
        adjust_chunks={'row': time_index.chunks[0]},
        meta=np.empty((0,)*len(vis_dims), dtype=out_dtype),
        dtype=out_dtype)

    for sb in range(1, src_blocks):
        base_vis = da.blockwise(
            _predict_coh_wrapper, vis_dims,
            time_index, ("row",),
            antenna1, ("row",),
            antenna2, ("row",),
            None if dde1_jones is None else dde1_jones.blocks[sb, ...],
            None if dde1_jones is None else ajones_dims,

            None if source_coh is None else source_coh.blocks[sb, ...],
            None if source_coh is None else src_coh_dims,

            None if dde2_jones is None else dde2_jones.blocks[sb, ...],
            None if dde2_jones is None else ajones_dims,

            None, None,
            base_vis, ("row", "chan") + cdims,
            None, None,
            reduce_source=True,
            # time+row dimension chunks are equivalent but differently sized
            align_arrays=False,
            # Force row dimension to take row chunking scheme,
            # instead of time chunking scheme
            adjust_chunks={'row': time_index.chunks[0]},
            meta=np.empty((0,)*len(vis_dims), dtype=out_dtype),
            dtype=out_dtype)

    return base_vis


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
    src_coh_dims = ("src", "row", "chan") + cdims

    coherencies = da.blockwise(
        _predict_coh_wrapper, src_coh_dims,
        time_index, ("row",),
        antenna1, ("row",),
        antenna2, ("row",),
        dde1_jones, None if dde1_jones is None else ajones_dims,
        source_coh, None if source_coh is None else src_coh_dims,
        dde2_jones, None if dde2_jones is None else ajones_dims,
        None, None,
        None, None,
        None, None,
        # time+row dimension chunks are equivalent but differently sized
        align_arrays=False,
        # Force row dimension to take row chunking scheme,
        # instead of time chunking scheme
        adjust_chunks={'row': time_index.chunks[0]},
        meta=np.empty((0,)*len(src_coh_dims), dtype=out_dtype),
        dtype=out_dtype)

    return coherencies.sum(axis=0)


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
    vis_dims = ("row", "chan") + cdims

    return da.blockwise(
        _predict_dies_wrapper, vis_dims,
        time_index, ("row",),
        antenna1, ("row",),
        antenna2, ("row",),
        None, None,
        None, None,
        None, None,
        die1_jones, None if die1_jones is None else gjones_dims,
        base_vis, None if base_vis is None else vis_dims,
        die2_jones, None if die2_jones is None else gjones_dims,
        # time+row dimension chunks are equivalent but differently sized
        align_arrays=False,
        # Force row dimension to take row chunking scheme,
        # instead of time chunking scheme
        adjust_chunks={'row': time_index.chunks[0]},
        dtype=out_dtype)


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
                                 for a in dtype_arrays
                                 if a is not None))

    # Apply direction dependent effects
    if have_coh or have_ddes:
        # We create separate graphs for computing coherencies and applying
        # the gains because coherencies are chunked over source which
        # must be summed and added to the (possibly present) base visibilities
        if streams is True:
            sum_coherencies = stream_reduction(time_index, antenna1, antenna2,
                                               dde1_jones, source_coh,
                                               dde2_jones, predict_check_tup,
                                               out_dtype)
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
streams : {False, True}
    If ``True`` the coherencies are serially summed in a linear chain.
    If ``False``, dask uses a tree style reduction algorithm.
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
