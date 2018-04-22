# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import wraps

from .phase import phase_delay_docs
from .phase import phase_delay as np_phase_delay
from .bright import brightness as np_brightness
from .parangles import parallactic_angles as np_parangles
from .feeds import feed_rotation as np_feed_rotation
from .transform import transform_sources as np_transform_sources
from .beam_cubes import beam_cube_dde as np_beam_cude_dde

from ..util import corr_shape as corr_shape_fn
from ..util.docs import on_rtd, doc_tuple_to_str, mod_docs
from ..util.requirements import have_packages, MissingPackageException

_package_requirements = ('dask.array',)
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

else:
    import numpy as np
    import dask.array as da

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
