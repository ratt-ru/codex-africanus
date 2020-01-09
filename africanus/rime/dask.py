# -*- coding: utf-8 -*-


from africanus.rime.phase import (phase_delay as np_phase_delay,
                                  PHASE_DELAY_DOCS)
from africanus.rime.parangles import parallactic_angles as np_parangles
from africanus.rime.feeds import feed_rotation as np_feed_rotation
from africanus.rime.feeds import FEED_ROTATION_DOCS
from africanus.rime.transform import transform_sources as np_transform_sources
from africanus.rime.fast_beam_cubes import (beam_cube_dde as np_beam_cube_dde,
                                            BEAM_CUBE_DOCS)
from africanus.rime.dask_predict import predict_vis, wsclean_predict  # noqa
from africanus.rime.zernike import zernike_dde as np_zernike_dde


from africanus.util.docs import mod_docs
from africanus.util.requirements import requires_optional
from africanus.util.type_inference import infer_complex_dtype

import numpy as np

try:
    import dask.array as da
except ImportError as e:
    da_import_error = e
else:
    da_import_error = None


def _phase_delay_wrap(lm, uvw, frequency, convention):
    return np_phase_delay(lm[0], uvw[0], frequency, convention=convention)


@requires_optional('dask.array', da_import_error)
def phase_delay(lm, uvw, frequency, convention='fourier'):
    """ Dask wrapper for phase_delay function """
    return da.core.blockwise(_phase_delay_wrap, ("source", "row", "chan"),
                             lm, ("source", "(l,m)"),
                             uvw, ("row", "(u,v,w)"),
                             frequency, ("chan",),
                             convention=convention,
                             dtype=infer_complex_dtype(lm, uvw, frequency))


def _parangle_wrapper(t, ap, fc, **kw):
    return np_parangles(t, ap[0], fc[0], **kw)


@requires_optional('dask.array', da_import_error)
def parallactic_angles(times, antenna_positions, field_centre, **kwargs):

    return da.core.blockwise(_parangle_wrapper, ("time", "ant"),
                             times, ("time",),
                             antenna_positions, ("ant", "xyz"),
                             field_centre, ("fc",),
                             dtype=times.dtype,
                             **kwargs)


@requires_optional('dask.array', da_import_error)
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

    return da.core.blockwise(np_feed_rotation, pa_dims + corr_dims,
                             parallactic_angles, pa_dims,
                             feed_type=feed_type,
                             new_axes={'corr-1': 2, 'corr-2': 2},
                             dtype=dtype)


def _xform_wrap(lm, parallactic_angles, pointing_errors,
                antenna_scaling, frequency, dtype_):
    return np_transform_sources(lm[0], parallactic_angles,
                                pointing_errors[0], antenna_scaling,
                                frequency, dtype=dtype_)


@requires_optional('dask.array', da_import_error)
def transform_sources(lm, parallactic_angles, pointing_errors,
                      antenna_scaling, frequency, dtype=None):

    if dtype is None:
        dtype = np.float64

    xform_inds = ("comp", "src", "time", "ant", "chan")

    return da.core.blockwise(_xform_wrap, xform_inds,
                             lm, ("src", "lm"),
                             parallactic_angles, ("time", "ant"),
                             pointing_errors, ("time", "ant", "lm"),
                             antenna_scaling, ("ant", "chan"),
                             frequency, ("chan",),
                             new_axes={"comp": 3},
                             dtype=dtype,
                             dtype_=dtype)


def _beam_cube_dde_wrapper(beam, beam_lm_extents, beam_freq_map,
                           lm, parallactic_angles,
                           point_errors, antenna_scaling,
                           frequencies):
    return np_beam_cube_dde(beam[0][0][0], beam_lm_extents[0][0],
                            beam_freq_map[0], lm[0],
                            parallactic_angles, point_errors[0],
                            antenna_scaling[0], frequencies)


@requires_optional('dask.array', da_import_error)
def beam_cube_dde(beam, beam_lm_extents, beam_freq_map,
                  lm, parallactic_angles,
                  point_errors, antenna_scaling,
                  frequencies):

    if not all(len(c) == 1 for c in beam.chunks):
        raise ValueError("Beam chunking unsupported")

    if not all(len(c) == 1 for c in beam_freq_map.chunks):
        raise ValueError("Beam frequency map chunking unsupported")

    if not all(len(c) == 1 for c in beam_lm_extents.chunks):
        raise ValueError("Chunking of beam_lm_extents unsupported")

    corr_shapes = beam.shape[3:]
    corr_dims = tuple("corr-%d" % i for i in range(len(corr_shapes)))

    dde_dims = ("source", "time", "ant", "chan") + corr_dims
    beam_dims = ("beam-lw", "beam-mh", "beam-nud") + corr_dims

    return da.core.blockwise(_beam_cube_dde_wrapper, dde_dims,
                             beam, beam_dims,
                             beam_lm_extents, ("beam-lm", "beam-ext"),
                             beam_freq_map, ("beam-nud",),
                             lm, ("source", "source-comp"),
                             parallactic_angles, ("time", "ant"),
                             point_errors, ("time", "ant", "chan", "pt-comp"),
                             antenna_scaling, ("ant", "chan", "scale-comp"),
                             frequencies, ("chan",),
                             dtype=beam.dtype)


def _zernike_wrapper(coords, coeffs, noll_index):
    # coords loses "three" dim
    # coeffs loses "poly" dim
    # noll_index loses "poly" dim
    return np_zernike_dde(coords[0], coeffs[0], noll_index[0])


@requires_optional('dask.array', da_import_error)
def zernike_dde(coords, coeffs, noll_index):
    ncorrs = len(coeffs.shape[2:-1])
    corr_dims = tuple("corr-%d" % i for i in range(ncorrs))

    return da.core.blockwise(_zernike_wrapper,
                             ("source", "time", "ant", "chan") + corr_dims,
                             coords,
                             ("three", "source", "time", "ant", "chan"),
                             coeffs,
                             ("ant", "chan") + corr_dims + ("poly",),
                             noll_index,
                             ("ant", "chan") + corr_dims + ("poly",),
                             dtype=coeffs.dtype)


try:
    phase_delay.__doc__ = PHASE_DELAY_DOCS.substitute(
                            array_type=":class:`dask.array.Array`")
except AttributeError:
    pass

try:
    parallactic_angles.__doc__ = mod_docs(np_parangles.__doc__,
                                          [(":class:`numpy.ndarray`",
                                            ":class:`dask.array.Array`")])
except AttributeError:
    pass

try:
    feed_rotation.__doc__ = FEED_ROTATION_DOCS.substitute(
                                array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass

try:
    transform_sources.__doc__ = mod_docs(np_transform_sources.__doc__,
                                         [(":class:`numpy.ndarray`",
                                           ":class:`dask.array.Array`")])
except AttributeError:
    pass

try:
    beam_cube_dde.__doc__ = BEAM_CUBE_DOCS.substitute(
                                array_type=":class:`dask.array.Array`")
except AttributeError:
    pass

try:
    zernike_dde.__doc__ = mod_docs(np_zernike_dde.__doc__,
                                   [(":class:`numpy.ndarray`",
                                     ":class:`dask.array.Array`")])
except AttributeError:
    pass
