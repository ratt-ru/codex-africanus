# -*- coding: utf-8 -*-

import numpy as np

from africanus.constants import c as lightspeed


def decorrelation(uvw, duvw_dtime, interval,
                  frequency, chan_width, lm,
                  time_smear=True, freq_smear=True):

    factor = np.ones((uvw.shape[0], chan_width.shape[0]), dtype=np.float64)
    n = np.sqrt(1.0 - lm[0]**2 - lm[1]**2) - 1.0

    # Frequency smearing
    if freq_smear:
        phase = uvw[:, 0]*lm[0] + uvw[:, 1]*lm[1] + uvw[:, 2]*n

        phi = np.pi * phase[:, None] * chan_width[None, :] / lightspeed
        non_zero = phi != 0.0
        phi = phi[non_zero]
        factor[non_zero] *= np.sin(phi)/phi

    # Smearing in time
    if time_smear:
        phase = (duvw_dtime[:, 0] * lm[0] +
                 duvw_dtime[:, 1] * lm[1] +
                 duvw_dtime[:, 2] * n) * interval

        phi = np.pi * phase[:, None] * frequency[None, :] / lightspeed
        non_zero = phi != 0.0
        phi = phi[non_zero]
        factor[non_zero] *= np.sin(phi)/phi

    return factor


def duvw_dtime(time, antenna1, antenna2, uvw):
    """
    Calculates dUVW / dTIME

    Parameters
    ----------
    time : :class:`numpy.ndarray`
        time in MJD seconds of shape :code:`(row,)`
    antenna1 : :class:`numpy.ndarray`
        antenna1 of shape :code:`(row,)`
    antenna2 : :class:`numpy.ndarray`
        antenna2 of shape :code;`(row,)`
    uvw : :class:`numpy.ndarray`
        uvw coordinates of shape :code:`(row, 3)`

    Returns
    -------
    duvw_dt : :class:`numpy.ndarray`
        change in uvw coordinates with respect to time
        of shape :code:`(row, 3)`.

    Notes
    -----

    1. Probably assumes monotically increasing time
    2. Takes a diff along UVW and TIME, for each baseline.
    3. Last row of the diff is replicated for the last input row.
       Does this have unintended side-effects?
    4. Baselines are grouped together with their mirrors
       for the purposes of this calculation.
    """

    # Copy antenna1 and antenna2, flipping mirror baselines
    # so that they're considered with their canonical brother.
    if not np.all(np.diff(time) >= 0.0):
        raise ValueError("time is not monotically increasing")

    ant1 = antenna1.copy()
    ant2 = antenna2.copy()

    mirror_bl = antenna2 < antenna1
    ant1[mirror_bl] = antenna2[mirror_bl]
    ant2[mirror_bl] = antenna1[mirror_bl]

    baselines = np.stack([ant1, ant2], axis=1)
    ubl, inv = np.unique(baselines, return_inverse=True, axis=0)

    # Generate a mask for each row containing the unique baseline
    bl_mask = np.arange(ubl.shape[0])[:, None] == inv[None, :]

    duvw_dt = np.zeros_like(uvw)

    # Handle each baseline
    for mask in bl_mask:
        time_sel = time[mask]
        uvw_sel = uvw[mask]

        uvw_diff = np.diff(uvw_sel, axis=0)
        time_diff = np.diff(time_sel)[:, None]

        # Divide change in uvw by change in time
        res = uvw_diff / time_diff
        # Assign the result, duplicating the last row's value
        duvw_dt[mask] = np.concatenate([res, res[-2:-1, :]])

    return duvw_dt
