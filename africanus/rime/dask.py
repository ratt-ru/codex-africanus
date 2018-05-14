# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
from functools import wraps

from .phase import phase_delay_docs
from .phase import phase_delay as np_phase_delay
from .bright import brightness as np_brightness
from .parangles import parallactic_angles as np_parangles
from .feeds import feed_rotation as np_feed_rotation
from .transform import transform_sources as np_transform_sources
from .beam_cubes import beam_cube_dde as np_beam_cude_dde
from .predict import predict_vis_docs
from .predict import predict_vis as np_predict_vis


from ..util.shapes import corr_shape as corr_shape_fn
from ..util.docs import on_rtd, doc_tuple_to_str, mod_docs
from ..util.requirements import have_packages, MissingPackageException

_package_requirements = ('dask.array', 'toolz')
have_requirements = have_packages(*_package_requirements)

if not have_requirements or on_rtd():
    def phase_delay(uvw, lm, frequency, dtype=None):
        raise MissingPackageException(*_package_requirements)

    def brightness(stokes, polarisation_type=None, corr_shape=None):
        raise MissingPackageException(*_package_requirements)

    def parallactic_angles(times, antenna_positions, field_centre, **kwargs):
        raise MissingPackageException(*_package_requirements)

    def feed_rotation(parallactic_angles, feed_type=None):
        raise MissingPackageException(*_package_requirements)

    def transform_sources(lm, parallactic_angles, pointing_errors,
                          antenna_scaling, dtype=None):
        raise MissingPackageException(*_package_requirements)

    def beam_cube_dde(beam, coords, l_grid, m_grid, freq_grid,
                      spline_order=1, mode='nearest'):
        raise MissingPackageException(*_package_requirements)

    def predict_vis(time_index, antenna1, antenna2,
                    ant1_jones, ant2_jones, row_jones,
                    g1_jones, g2_jones):
        raise MissingPackageException(*_package_requirements)

else:
    import numpy as np
    import dask.array as da

    try:
        import cytoolz as toolz
    except ImportError:
        import toolz

    def phase_delay(uvw, lm, frequency, dtype=np.complex128):
        """ Dask wrapper for phase_delay function """
        @wraps(np_phase_delay)
        def _wrapper(uvw, lm, frequency, dtype_):
            return np_phase_delay(uvw[0], lm[0], frequency, dtype=dtype_)

        return da.core.atop(_wrapper, ("source", "row", "chan"),
                            uvw, ("row", "(u,v,w)"),
                            lm, ("source", "(l,m)"),
                            frequency, ("chan",),
                            dtype=dtype,
                            dtype_=dtype)

    def brightness(stokes, polarisation_type=None, corr_shape=None):
        if corr_shape is None:
            corr_shape = 'flat'

        # Separate shape into head and tail
        head, npol = stokes.shape[:-1], stokes.shape[-1]

        if not npol == stokes.chunks[-1][0]:
            raise ValueError("The polarisation dimension "
                             "of the 'stokes' array "
                             "may not be chunked "
                             "(the chunk size must match "
                             "the dimension size).")

        # Create unique strings for the head dimensions
        head_dims = tuple("head-%d" % i for i in range(len(head)))
        # Figure out what our correlation shape should look like
        corr_shapes = corr_shape_fn(npol, corr_shape)
        # Create unique strings for the correlation dimensions
        corr_dims = tuple("corr-%d" % i for i in range(len(corr_shapes)))
        # We're introducing new axes for the correlations
        # with a fixed shape
        new_axes = {d: s for d, s in zip(corr_dims, corr_shapes)}

        @wraps(np_brightness)
        def _wrapper(stokes):
            return np_brightness(stokes[0],
                                 polarisation_type=polarisation_type,
                                 corr_shape=corr_shape)

        return da.core.atop(_wrapper, head_dims + corr_dims,
                            stokes, head_dims + ("pol",),
                            new_axes=new_axes,
                            dtype=np.complex64 if stokes.dtype == np.float32
                            else np.complex128)

    def parallactic_angles(times, antenna_positions, field_centre, **kwargs):
        @wraps(np_parangles)
        def _wrapper(t, ap, fc, **kw):
            return np_parangles(t, ap[0], fc[0], **kwargs)

        return da.core.atop(_wrapper, ("time", "ant"),
                            times, ("time",),
                            antenna_positions, ("ant", "xyz"),
                            field_centre, ("fc",),
                            dtype=times.dtype,
                            **kwargs)

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

    def transform_sources(lm, parallactic_angles, pointing_errors,
                          antenna_scaling, frequency, dtype=None):

        @wraps(np_transform_sources)
        def _wrapper(lm, parallactic_angles, pointing_errors,
                     antenna_scaling, frequency, dtype_):
            return np_transform_sources(lm[0], parallactic_angles,
                                        pointing_errors[0], antenna_scaling,
                                        frequency, dtype=dtype_)

        if dtype is None:
            dtype = np.float64

        return da.core.atop(_wrapper, ("comp", "src", "time", "ant", "chan"),
                            lm, ("src", "lm"),
                            parallactic_angles, ("time", "ant"),
                            pointing_errors, ("time", "ant", "lm"),
                            antenna_scaling, ("ant", "chan"),
                            frequency, ("chan",),
                            new_axes={"comp": 3},
                            dtype=dtype,
                            dtype_=dtype)

    def beam_cube_dde(beam, coords, l_grid, m_grid, freq_grid,
                      spline_order=1, mode='nearest'):

        @wraps(np_beam_cude_dde)
        def _wrapper(beam, coords, l_grid, m_grid, freq_grid,
                     spline_order=1, mode='nearest'):
            return np_beam_cude_dde(beam[0][0][0], coords[0],
                                    l_grid[0], m_grid[0], freq_grid[0],
                                    spline_order=spline_order, mode=mode)

        coord_shapes = coords.shape[1:]
        corr_shapes = beam.shape[3:]
        corr_dims = tuple("corr-%d" % i for i in range(len(corr_shapes)))
        coord_dims = tuple("coord-%d" % i for i in range(len(coord_shapes)))

        beam_dims = ("beam_lw", "beam_mh", "beam_nud") + corr_dims

        return da.core.atop(_wrapper, coord_dims + corr_dims,
                            beam, beam_dims,
                            coords, ("coords",) + coord_dims,
                            l_grid, ("beam_lw",),
                            m_grid, ("beam_mh",),
                            freq_grid, ("beam_nud",),
                            spline_order=spline_order,
                            mode=mode,
                            dtype=beam.dtype)

    def predict_vis(time_index, antenna1, antenna2,
                    ant1_jones, ant2_jones, row_jones,
                    g1_jones, g2_jones):

        @wraps(np_predict_vis)
        def _wrapper(time_index, antenna1, antenna2,
                     ant1_jones, ant2_jones, row_jones,
                     g1_jones, g2_jones):

            # Normalise the time index
            time_index -= time_index.min()

            return np_predict_vis(time_index, antenna1, antenna2,
                                  ant1_jones[0][0], ant2_jones[0][0],
                                  row_jones[0], g1_jones[0], g2_jones[0])

        if ant1_jones.shape[2] != ant1_jones.chunks[2][0]:
            raise ValueError("Subdivision of antenna dimension into "
                             "multiple chunks is not supported.")

        if len(ant1_jones.chunks[1]) != len(time_index.chunks[0]):
            raise ValueError("Number of row chunks (%s) does not equal "
                             "number of time chunks (%s)." %
                             (time_index.chunks[0], ant1_jones.chunks[1]))

        # Generate strings for the correlation dimensions
        cdims = tuple("corr-%d" % i for i in range(len(row_jones.shape[3:])))
        ajones_dims = ("src", "row", "ant", "chan") + cdims

        # In the case predict_vis, the "row" and "time" dimensions
        # are intimately related -- a contiguous series of rows
        # are related to a contiguous series of timesteps.
        # This means that the number of chunks of these
        # two dimensions must match even though the chunk sizes may not.
        # da.core.atop insists on matching chunk sizes.
        # For this reason, we use the lower level da.core.top and
        # substitute "row" for "time" in arrays such as ant1_jones
        # and g1_jones.
        token = da.core.tokenize(time_index, antenna1, antenna2,
                                 ant1_jones, ant2_jones, row_jones,
                                 g1_jones, g2_jones)
        name = "-".join(("predict_vis", token))
        dsk = da.core.top(_wrapper, name, ("row", "chan") + cdims,
                          time_index.name, ("row",),
                          antenna1.name, ("row",),
                          antenna2.name, ("row",),
                          ant1_jones.name, ajones_dims,
                          ant2_jones.name, ajones_dims,
                          row_jones.name, ("src", "row", "chan") + cdims,
                          g1_jones.name, ("row", "ant", "chan") + cdims,
                          g2_jones.name, ("row", "ant", "chan") + cdims,
                          numblocks={
                                time_index.name: time_index.numblocks,
                                antenna1.name: antenna1.numblocks,
                                antenna2.name: antenna2.numblocks,
                                ant1_jones.name: ant1_jones.numblocks,
                                ant2_jones.name: ant2_jones.numblocks,
                                row_jones.name: row_jones.numblocks,
                                g1_jones.name: g1_jones.numblocks,
                                g2_jones.name: g2_jones.numblocks,
                            })

        # Merge input graphs into the top graph
        dsk = toolz.merge(dsk, *(a.__dask_graph__() for a in (time_index,
                                                              antenna1,
                                                              antenna2,
                                                              ant1_jones,
                                                              ant2_jones,
                                                              row_jones,
                                                              g1_jones,
                                                              g2_jones)))

        # We can infer output chunk sizes from row_jones
        chunks = row_jones.chunks[1:]

        return da.Array(dsk, name, chunks, dtype=ant1_jones.dtype)

phase_delay.__doc__ = doc_tuple_to_str(phase_delay_docs,
                                       [(":class:`numpy.ndarray`",
                                         ":class:`dask.array.Array`")])

brightness.__doc__ = mod_docs(np_brightness.__doc__,
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


dask_mp_docs = OrderedDict((k, getattr(predict_vis_docs, k)) for k
                           in predict_vis_docs._fields)

dask_mp_docs['notes'] += (
    """* The ``ant`` dimension should only contain a single chunk equal
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
""")


predict_vis.__doc__ = doc_tuple_to_str(dask_mp_docs,
                                       [(":class:`numpy.ndarray`",
                                         ":class:`dask.array.Array`"),
                                        (":func:`~numpy.einsum`",
                                         ":func:`~dask.array.einsum`")])
