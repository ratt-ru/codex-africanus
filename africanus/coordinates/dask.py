# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numba
import numpy as np

try:
    import dask.array as da
except ImportError:
    pass


from ..util.requirements import requires_optional
from .coordinates import (radec_to_lmn as np_radec_to_lmn,
                          lmn_to_radec as np_lmn_to_radec,
                          RADEC_TO_LMN_DOCS,
                          LMN_TO_RADEC_DOCS)


@requires_optional('dask.array')
def radec_to_lmn(radec, phase_centre=None):
    phase_centre_dims = ("(l,m)",) if phase_centre is not None else None

    return da.core.atop(np_radec_to_lmn, ("source", "(l,m)"),
                        radec, ("source", "(l,m)"),
                        phase_centre, phase_centre_dims,
                        dtype=radec.dtype)


@requires_optional('dask.array')
def lmn_to_radec(lmn, phase_centre=None):
    phase_centre_dims = ("(l,m)",) if phase_centre is not None else None

    return da.core.atop(np_lmn_to_radec, ("source", "(l,m)"),
                        lmn, ("source", "(l,m)"),
                        phase_centre, phase_centre_dims,
                        dtype=lmn.dtype)


try:
    radec_to_lmn.__doc__ = RADEC_TO_LMN_DOCS.substitute(
                                array_type=":class:`dask.array.Array`")
    lmn_to_radec.__doc__ = LMN_TO_RADEC_DOCS.substitute(
                                array_type=":class:`dask.array.Array`")
except AttributeError:
    pass
