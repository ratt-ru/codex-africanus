# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
from functools import wraps

from .phase import phase_delay_docs
from .phase import phase_delay as np_phase_delay
from .parangles import parallactic_angles as np_parangles
from .feeds import feed_rotation as np_feed_rotation
from .transform import transform_sources as np_transform_sources
from .beam_cubes import beam_cube_dde as np_beam_cude_dde
from .predict import PREDICT_DOCS
from .predict import predict_vis as np_predict_vis
from .zernike import zernike_dde as np_zernike_dde


from ..util.docs import doc_tuple_to_str, mod_docs
from ..util.requirements import requires_optional

import numpy as np

try:
    import dask.array as da
    from dask.sharedict import ShareDict
except ImportError:
    pass

try:
    import cytoolz as toolz
except ImportError:
    try:
        import toolz
    except ImportError:
        pass


@wraps(np_phase_delay)
def _phase_delay_wrap(uvw, lm, frequency, dtype_):
    return np_phase_delay(uvw[0], lm[0], frequency, dtype=dtype_)


@requires_optional('dask.array')
def phase_delay(uvw, lm, frequency, dtype=np.complex128):
    """ Dask wrapper for phase_delay function """
    return da.core.atop(_phase_delay_wrap, ("source", "row", "chan"),
                        uvw, ("row", "(u,v,w)"),
                        lm, ("source", "(l,m)"),
                        frequency, ("chan",),
                        dtype=dtype,
                        dtype_=dtype)


@wraps(np_parangles)
def _parangle_wrapper(t, ap, fc, **kw):
    return np_parangles(t, ap[0], fc[0], **kw)


@requires_optional('dask.array')
def parallactic_angles(times, antenna_positions, field_centre, **kwargs):

    return da.core.atop(_parangle_wrapper, ("time", "ant"),
                        times, ("time",),
                        antenna_positions, ("ant", "xyz"),
                        field_centre, ("fc",),
                        dtype=times.dtype,
                        **kwargs)


@requires_optional('dask.array')
def feed_rotation(parallactic_angles, feed_type):
    pa_dims = tuple("pa-%d" % i for i in range(parallactic_angles.ndim))
    corr_dims = ('corr-1', 'corr-2')

    if parallactic_angles.dtype == np.float32:
        dtype = np.complex64
    elif parallactic_angles.dtype == np.float64:
        dtype = np.complex128
    else:
        raise ValueError("parallactic_angles have "
                         "non-floating point dtype")

    return da.core.atop(np_feed_rotation, pa_dims + corr_dims,
                        parallactic_angles, pa_dims,
                        feed_type=feed_type,
                        new_axes={'corr-1': 2, 'corr-2': 2},
                        dtype=dtype)


@wraps(np_transform_sources)
def _xform_wrap(lm, parallactic_angles, pointing_errors,
                antenna_scaling, frequency, dtype_):
    return np_transform_sources(lm[0], parallactic_angles,
                                pointing_errors[0], antenna_scaling,
                                frequency, dtype=dtype_)


@requires_optional('dask.array')
def transform_sources(lm, parallactic_angles, pointing_errors,
                      antenna_scaling, frequency, dtype=None):

    if dtype is None:
        dtype = np.float64

    return da.core.atop(_xform_wrap, ("comp", "src", "time", "ant", "chan"),
                        lm, ("src", "lm"),
                        parallactic_angles, ("time", "ant"),
                        pointing_errors, ("time", "ant", "lm"),
                        antenna_scaling, ("ant", "chan"),
                        frequency, ("chan",),
                        new_axes={"comp": 3},
                        dtype=dtype,
                        dtype_=dtype)


@wraps(np_beam_cude_dde)
def _beam_wrapper(beam, coords, l_grid, m_grid, freq_grid,
                  spline_order=1, mode='nearest'):
    return np_beam_cude_dde(beam[0][0][0], coords[0],
                            l_grid[0], m_grid[0], freq_grid[0],
                            spline_order=spline_order, mode=mode)


@requires_optional('dask.array')
def beam_cube_dde(beam, coords, l_grid, m_grid, freq_grid,
                  spline_order=1, mode='nearest'):

    coord_shapes = coords.shape[1:]
    corr_shapes = beam.shape[3:]
    corr_dims = tuple("corr-%d" % i for i in range(len(corr_shapes)))
    coord_dims = tuple("coord-%d" % i for i in range(len(coord_shapes)))

    beam_dims = ("beam_lw", "beam_mh", "beam_nud") + corr_dims

    return da.core.atop(_beam_wrapper, coord_dims + corr_dims,
                        beam, beam_dims,
                        coords, ("coords",) + coord_dims,
                        l_grid, ("beam_lw",),
                        m_grid, ("beam_mh",),
                        freq_grid, ("beam_nud",),
                        spline_order=spline_order,
                        mode=mode,
                        dtype=beam.dtype)


@wraps(np_zernike_dde)
def _zernike_wrapper(coords, coeffs, noll_index):
    # coords loses "three" dim
    # coeffs loses "poly" dim
    # noll_index loses "poly" dim
    return np_zernike_dde(coords[0], coeffs[0], noll_index[0])


@requires_optional('dask.array')
def zernike_dde(coords, coeffs, noll_index):
    ncorrs = len(coeffs.shape[2:-1])
    corr_dims = tuple("corr-%d" % i for i in range(ncorrs))

    return da.core.atop(_zernike_wrapper,
                        ("source", "time", "ant", "chan") + corr_dims,
                        coords,
                        ("three", "source", "time", "ant", "chan"),
                        coeffs,
                        ("ant", "chan") + corr_dims + ("poly",),
                        noll_index,
                        ("ant", "chan") + corr_dims + ("poly",),
                        dtype=coeffs.dtype)


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


@requires_optional('dask.array')
def predict_vis(time_index, antenna1, antenna2,
                dde1_jones=None, source_coh=None, dde2_jones=None,
                die1_jones=None, base_vis=None, die2_jones=None):

    have_a1 = dde1_jones is not None
    have_a2 = dde2_jones is not None
    have_bl = source_coh is not None
    have_g1 = die1_jones is not None
    have_coh = base_vis is not None
    have_g2 = die2_jones is not None

    if have_a1 ^ have_a2:
        raise ValueError("Both dde1_jones and dde2_jones "
                         "must be present or absent")

    have_ants = have_a1 and have_a2

    if have_ants:
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

    if have_g1 ^ have_g2:
        raise ValueError("Both die1_jones and die2_jones "
                         "must be present or absent")

    have_dies = have_g1 and have_g2

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

    # Generate strings for the correlation dimensions
    if have_ants:
        cdims = tuple("corr-%d" % i for i in range(len(dde1_jones.shape[4:])))
    elif have_bl:
        cdims = tuple("corr-%d" % i for i in range(len(source_coh.shape[3:])))
    elif have_dies:
        cdims = tuple("corr-%d" % i for i in range(len(die1_jones.shape[3:])))
    else:
        raise ValueError("Missing both antenna and baseline jones terms")

    # Infer the output dtype
    dtype_arrays = [dde1_jones, source_coh, dde2_jones, die1_jones, die2_jones]
    out_dtype = np.result_type(*(np.dtype(a.dtype.name)
                                 for a in dtype_arrays if a is not None))

    # In the case of predict_vis, the "row" and "time" dimensions
    # are intimately related -- a contiguous series of rows
    # are related to a contiguous series of timesteps.
    # This means that the number of chunks of these
    # two dimensions must match even though the chunk sizes may not.
    # da.core.atop insists on matching chunk sizes.
    # For this reason, we use the lower level da.core.top and
    # substitute "row" for "time" in arrays such as dde1_jones
    # and die1_jones.
    token = da.core.tokenize(time_index, antenna1, antenna2,
                             dde1_jones, source_coh, dde2_jones, base_vis)

    ajones_dims = ("src", "row", "ant", "chan") + cdims
    gjones_dims = ("row", "ant", "chan") + cdims

    # Setup
    # 1. Optional top arguments
    # 2. Optional numblocks kwarg
    # 3. dask graph inputs
    array_dsk = ShareDict()
    top_args = [time_index.name, ("row",),
                antenna1.name, ("row",),
                antenna2.name, ("row",)]
    numblocks = {
        time_index.name: time_index.numblocks,
        antenna1.name: antenna1.numblocks,
        antenna2.name: antenna2.numblocks
    }

    # Merge input graphs into the top graph
    array_dsk.update(time_index.__dask_graph__())
    array_dsk.update(antenna1.__dask_graph__())
    array_dsk.update(antenna2.__dask_graph__())

    # Handle presence/absence of dde1_jones
    if have_ants:
        top_args.extend([dde1_jones.name, ajones_dims])
        numblocks[dde1_jones.name] = dde1_jones.numblocks
        array_dsk.update(dde1_jones.__dask_graph__())
        other_chunks = dde1_jones.chunks[3:]
        src_chunks = dde1_jones.chunks[0]
    else:
        top_args.extend([None, None])

    # Handle presence/absence of source_coh
    if have_bl:
        top_args.extend([source_coh.name, ("src", "row", "chan") + cdims])
        numblocks[source_coh.name] = source_coh.numblocks
        other_chunks = source_coh.chunks[2:]
        src_chunks = source_coh.chunks[0]
        array_dsk.update(source_coh.__dask_graph__())
    else:
        top_args.extend([None, None])

    # Handle presence/absence of dde2_jones
    if have_ants:
        top_args.extend([dde2_jones.name, ajones_dims])
        numblocks[dde2_jones.name] = dde2_jones.numblocks
        other_chunks = dde1_jones.chunks[3:]
        array_dsk.update(dde2_jones.__dask_graph__())
        other_chunks = dde2_jones.chunks[3:]
        src_chunks = dde1_jones.chunks[0]
    else:
        top_args.extend([None, None])

    # die1_jones, base_vis and die2_jones absent for this part of the graph
    top_args.extend([None, None, None, None, None, None])

    assert len(top_args) // 2 == 9, len(top_args) // 2

    name = "-".join(("predict_vis", token))
    dsk = da.core.top(_predict_coh_wrapper,
                      name, ("src", "row", "chan") + cdims,
                      *top_args,
                      numblocks=numblocks)

    array_dsk.update(dsk)

    # We can infer output chunk sizes from source_coh
    chunks = ((1,)*len(src_chunks), time_index.chunks[0],) + other_chunks

    sum_coherencies = da.Array(array_dsk, name, chunks, dtype=out_dtype)
    sum_coherencies = sum_coherencies.sum(axis=0)

    if have_coh:
        sum_coherencies += base_vis

    if not have_dies:
        return sum_coherencies

    # Now apply any Direction Independent Effect Terms

    # Setup
    # 1. Optional top arguments
    # 2. Optional numblocks kwarg
    # 3. dask graph inputs
    array_dsk = ShareDict()
    top_args = [time_index.name, ("row",),
                antenna1.name, ("row",),
                antenna2.name, ("row",)]
    numblocks = {
        time_index.name: time_index.numblocks,
        antenna1.name: antenna1.numblocks,
        antenna2.name: antenna2.numblocks
    }

    array_dsk.update(time_index.__dask_graph__())
    array_dsk.update(antenna1.__dask_graph__())
    array_dsk.update(antenna2.__dask_graph__())

    # dde1_jones, source_coh  and dde2_jones not present
    top_args.extend([None, None, None, None, None, None])

    top_args.extend([die1_jones.name, gjones_dims])
    top_args.extend([sum_coherencies.name, ("row", "chan") + cdims])
    top_args.extend([die2_jones.name, gjones_dims])
    numblocks[die1_jones.name] = die1_jones.numblocks
    numblocks[sum_coherencies.name] = sum_coherencies.numblocks
    numblocks[die2_jones.name] = die2_jones.numblocks
    array_dsk.update(die1_jones.__dask_graph__())
    array_dsk.update(sum_coherencies.__dask_graph__())
    array_dsk.update(die2_jones.__dask_graph__())

    assert len(top_args) // 2 == 9

    token = da.core.tokenize(time_index, antenna1, antenna2,
                             die1_jones, sum_coherencies, die2_jones)
    name = '-'.join(("predict_vis", token))
    dsk = da.core.top(_predict_dies_wrapper,
                      name, ("row", "chan") + cdims,
                      *top_args, numblocks=numblocks)
    array_dsk.update(dsk)

    chunks = (time_index.chunks[0],) + other_chunks

    return da.Array(array_dsk, name, chunks, dtype=out_dtype)


phase_delay.__doc__ = doc_tuple_to_str(phase_delay_docs,
                                       [(":class:`numpy.ndarray`",
                                         ":class:`dask.array.Array`")])

parallactic_angles.__doc__ = mod_docs(np_parangles.__doc__,
                                      [(":class:`numpy.ndarray`",
                                        ":class:`dask.array.Array`")])

feed_rotation.__doc__ = mod_docs(np_feed_rotation.__doc__,
                                 [(":class:`numpy.ndarray`",
                                   ":class:`dask.array.Array`")])

transform_sources.__doc__ = mod_docs(np_transform_sources.__doc__,
                                     [(":class:`numpy.ndarray`",
                                       ":class:`dask.array.Array`")])

beam_cube_dde.__doc__ = mod_docs(np_beam_cude_dde.__doc__,
                                 [(":class:`numpy.ndarray`",
                                   ":class:`dask.array.Array`")])

zernike_dde.__doc__ = mod_docs(np_zernike_dde.__doc__,
                               [(":class:`numpy.ndarray`",
                                   ":class:`dask.array.Array`")])


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
                                extra_notes=EXTRA_DASK_NOTES)
except AttributeError:
    pass
