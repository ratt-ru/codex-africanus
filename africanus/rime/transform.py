#!/usr/bin/env python
# -*- coding: utf-8 -*-


import math

import numpy as np

from africanus.util.numba import jit


@jit(nopython=True, nogil=True, cache=True)
def _nb_transform_sources(lm, parallactic_angles, pointing_errors,
                          antenna_scaling, frequency, coords):
    """
    numba implementation of
    :func:`~africanus.rime.transform_sources`
    """
    _, nsrc, ntime, na, nchan = coords.shape

    for t in range(ntime):
        for a in range(na):
            pa_sin = math.sin(parallactic_angles[t, a])
            pa_cos = math.cos(parallactic_angles[t, a])

            for s in range(nsrc):
                l, m = lm[s]

                # Rotate source coordinate by parallactic angle
                l = l*pa_cos - m*pa_sin  # noqa
                m = l*pa_sin + m*pa_cos

                # Add pointing errors
                l += pointing_errors[t, a, 0]  # noqa
                m += pointing_errors[t, a, 1]

                # Scale by antenna scaling factors
                for c in range(nchan):
                    coords[0, s, t, a, c] = l*antenna_scaling[a, c]
                    coords[1, s, t, a, c] = m*antenna_scaling[a, c]
                    coords[2, s, t, a, c] = frequency[c]

    return coords


def transform_sources(lm, parallactic_angles, pointing_errors,
                      antenna_scaling, frequency, dtype=None):
    """
    Creates beam sampling coordinates suitable for use
    in :func:`~africanus.rime.beam_cube_dde` by:

    1. Rotating ``lm`` coordinates by the ``parallactic_angles``
    2. Adding ``pointing_errors``
    3. Scaling by ``antenna_scaling``

    Parameters
    ----------
    lm : :class:`numpy.ndarray`
        LM coordinates of shape :code:`(src,2)` in radians
        offset from the phase centre.
    parallactic_angles : :class:`numpy.ndarray`
        parallactic angles of shape :code:`(time, antenna)`
        in radians.
    pointing_errors : :class:`numpy.ndarray`
        LM pointing errors for each antenna at
        each timestep in radians.
        Has shape :code:`(time, antenna, 2)`
    antenna_scaling : :class:`numpy.ndarray`
        antenna scaling factor for each channel and
        each antenna. Has shape :code:`(antenna, chan)`
    frequency : :class:`numpy.ndarray`
        frequencies for each channel. Has shape :code:`(chan,)`
    dtype : :class:`numpy.dtype`, optional
        Numpy dtype of result array. Should be float32 or float64.
        Defaults to float64


    Returns
    -------
    coords : :class:`numpy.ndarray`
        coordinates of shape :code:`(3, src, time, antenna, chan)`
        where each coordinate component represents **l**, **m** and
        **frequency**, respectively.
    """

    ntime, na = parallactic_angles.shape
    nsrc = lm.shape[0]
    assert (ntime, na, 2) == pointing_errors.shape
    nchan = antenna_scaling.shape[1]
    assert nchan == frequency.shape[0]

    dtype = np.float64 if dtype is None else dtype
    coords = np.empty((3, nsrc, ntime, na, nchan), dtype=dtype)

    return _nb_transform_sources(lm, parallactic_angles, pointing_errors,
                                 antenna_scaling, frequency, coords)
