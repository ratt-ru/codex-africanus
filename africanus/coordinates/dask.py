# -*- coding: utf-8 -*-


try:
    import dask.array as da
except ImportError as e:
    dask_import_error = e
else:
    dask_import_error = None


from africanus.util.requirements import requires_optional
from africanus.coordinates.coordinates import (radec_to_lmn as np_radec_to_lmn,
                                               radec_to_lm as np_radec_to_lm,
                                               lmn_to_radec as np_lmn_to_radec,
                                               lm_to_radec as np_lm_to_radec,
                                               RADEC_TO_LMN_DOCS,
                                               LMN_TO_RADEC_DOCS)


def _radec_to_lmn(radec, phase_centre):
    return np_radec_to_lmn(radec[0], phase_centre[0] if phase_centre else None)


@requires_optional('dask.array', dask_import_error)
def radec_to_lmn(radec, phase_centre=None):
    phase_centre_dims = ("radec",) if phase_centre is not None else None

    return da.core.blockwise(_radec_to_lmn, ("source", "lmn"),
                             radec, ("source", "radec"),
                             phase_centre, phase_centre_dims,
                             new_axes={"lmn": 3},
                             dtype=radec.dtype)


def _lmn_to_radec(lmn, phase_centre):
    return np_lmn_to_radec(lmn[0], phase_centre)


@requires_optional('dask.array', dask_import_error)
def lmn_to_radec(lmn, phase_centre=None):
    phase_centre_dims = ("radec",) if phase_centre is not None else None

    return da.core.blockwise(_lmn_to_radec, ("source", "radec"),
                             lmn, ("source", "lmn"),
                             phase_centre, phase_centre_dims,
                             new_axes={"radec": 2},
                             dtype=lmn.dtype)


def _radec_to_lm(radec, phase_centre):
    return np_radec_to_lm(radec[0], phase_centre[0] if phase_centre else None)


@requires_optional('dask.array', dask_import_error)
def radec_to_lm(radec, phase_centre=None):
    phase_centre_dims = ("radec",) if phase_centre is not None else None

    return da.core.blockwise(_radec_to_lm, ("source", "lm"),
                             radec, ("source", "radec"),
                             phase_centre, phase_centre_dims,
                             new_axes={"lm": 2},
                             dtype=radec.dtype)


def _lm_to_radec(lm, phase_centre):
    return np_lm_to_radec(lm[0], phase_centre)


@requires_optional('dask.array', dask_import_error)
def lm_to_radec(lm, phase_centre=None):
    phase_centre_dims = ("radec",) if phase_centre is not None else None

    return da.core.blockwise(_lm_to_radec, ("source", "radec"),
                             lm, ("source", "lm"),
                             phase_centre, phase_centre_dims,
                             new_axes={"radec": 2},
                             dtype=lm.dtype)


try:
    radec_to_lmn.__doc__ = RADEC_TO_LMN_DOCS.substitute(
                                lm_components="3",
                                array_type=":class:`dask.array.Array`")
    radec_to_lm.__doc__ = RADEC_TO_LMN_DOCS.substitute(
                                lm_components="2",
                                array_type=":class:`dask.array.Array`")
    lmn_to_radec.__doc__ = LMN_TO_RADEC_DOCS.substitute(
                                lm_name="lmn", lm_components="3",
                                array_type=":class:`dask.array.Array`")
    lm_to_radec.__doc__ = LMN_TO_RADEC_DOCS.substitute(
                                lm_name="lm", lm_components="2",
                                array_type=":class:`dask.array.Array`")


except AttributeError:
    pass
