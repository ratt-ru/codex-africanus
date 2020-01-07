# -*- coding: utf-8 -*-

"""
References

Synthesis Imaging in Radio Astronomy II
Lecture 18. Bandwidth and Time-Average Smearing.
"""

import numpy as np

from africanus.constants import c


# Δ ∇ Ψ ɸ
# ν


def decorrelation_map(time, uvw, ant1, ant2,
                      chan_freq, chan_width,
                      a1, a2,
                      l_max=1, decorrelation=0.98):

    rows = np.logical_and(a1 == ant1, a2 == ant2)
    uvw = uvw[rows]
    time = time[rows]
    nrows = uvw.shape[0]
    nchan = chan_freq.shape[0]

    # Calculate delta UVW
    Δuvw = np.zeros_like(uvw)
    Δuvw[:-1, :] = uvw[:-1, :] - uvw[1:, :]
    # last delta copied from previous row
    Δuvw[-1, :] = Δuvw[-2, :]

    # ΔΨ Tolerance. Derived from
    # Equation 18-31 in Synthesis Imaging
    ΔΨ = np.sqrt(6. * (1.0 - decorrelation))

    # max delta phase occurs when Δuvw lines up with lmn - 1.
    # So assume we have an lmn vector such
    # that ||(l,m)||=l_max, n_max=|sqrt(1 - l_max^2) - 1|
    nu_max = chan_freq.max()
    Δuv = c * ΔΨ / (2 * np.pi * nu_max)
    n_max = np.abs(np.sqrt(1.0 - l_max**2) - 1.0)
    # the max phase delta for each row will be ||(Δu,Δv)||*l_max + |Δw|*n_max
    Δɸ = ((2.0 * np.pi * nu_max / c) *
            (np.sqrt((Δuvw[:, :2]**2).sum(axis=1))*l_max +
             np.abs(Δuvw[:, 2])*n_max))

    # max phase \vec{u}\cdot\vec{l} is ||(u,v)||*l_max +|w|*n_max
    ɸ_max = np.sqrt((uvw[:, :2]**2).sum(axis=1))*l_max + abs(uvw[:, 2])*n_max
    # delta nu for each row. Corresponds to phase change of ΔΨ
    Δnu = (c / (2.0 * np.pi)) * (ΔΨ / ɸ_max)

    block_slices = []
    start = 0
    row_Δɸ_sum = 0.0

    # row_Δɸ[i] is distance of row i+1 wrt row i
    for row, (row_Δɸ, row_Δnu) in enumerate(zip(Δɸ, Δnu)):
        row_Δɸ_sum += row_Δɸ

        # if more than critical, then block is [start,row + 1)
        if row_Δɸ_sum > ΔΨ:
            block_slices.append(slice(start, row + 1))
            row_Δɸ_sum = 0.0
            start = row + 1

    # add last block
    if start < nrows:
        block_slices.append(slice(start, nrows))

    return

    print(block_slices)

    frac_bandwidth = chan_width / chan_width.sum()

    print(Δnu)

    frac_chan_block_size = Δnu[:, None] / chan_width[None, :]
    print(frac_chan_block_size)
    # frac_chan_block_size = Δnu / chan_width[0]
    frac_chan_block_min = np.array([max(frac_chan_block_size[slc].min(), 1)
                                    for slc in block_slices])

    num_chan_blocks = np.ceil(nchan / frac_chan_block_min)

    # per each time block
    size_chan_block = np.int32(np.ceil(nchan / num_chan_blocks))

    unique_chan_block_sizes = np.unique(size_chan_block)
    nuniq_blocks = unique_chan_block_sizes.shape[0]

    chan_range = np.arange(nchan, dtype=np.int32)
    chan_pairs = np.zeros((nuniq_blocks, nchan + 1, 2), dtype=np.intp)
    chan_pairs[:, :-1, 0] = chan_range[None, :] / unique_chan_block_sizes[:, None]
    chan_pairs[:, :-1, 1] = chan_range[None, :] / unique_chan_block_sizes[:, None]

    print(frac_chan_block_min)
    print(num_chan_blocks)
    print(size_chan_block)
    print(unique_chan_block_sizes)


def decorrelation(uvw, Δuvw_Δtime, interval,
                  frequency, chan_width, lm,
                  time_smear=True, freq_smear=True):

    factor = np.ones((uvw.shape[0], chan_width.shape[0]), dtype=np.float64)
    n = np.sqrt(1.0 - lm[0]**2 - lm[1]**2) - 1.0

    # Frequency smearing
    if freq_smear:
        phase = uvw[:, 0]*lm[0] + uvw[:, 1]*lm[1] + uvw[:, 2]*n

        phi = np.pi * phase[:, None] * chan_width[None, :] / c
        non_zero = phi != 0.0
        phi = phi[non_zero]
        factor[non_zero] *= np.sin(phi)/phi

    # Smearing in time
    if time_smear:
        phase = (Δuvw_Δtime[:, 0] * lm[0] +
                 Δuvw_Δtime[:, 1] * lm[1] +
                 Δuvw_Δtime[:, 2] * n) * interval

        phi = np.pi * phase[:, None] * frequency[None, :] / c
        non_zero = phi != 0.0
        phi = phi[non_zero]
        factor[non_zero] *= np.sin(phi)/phi

    return factor


def Δuvw_Δtime(time, antenna1, antenna2, uvw):
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
    Δuvw_dt : :class:`numpy.ndarray`
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

    Δuvw_dt = np.zeros_like(uvw)

    # Handle each baseline
    for bl in range(ubl.shape[0]):
        mask = bl == inv
        time_sel = time[mask]
        uvw_sel = uvw[mask]

        uvw_diff = np.diff(uvw_sel, axis=0)
        time_diff = np.diff(time_sel)[:, None]

        # Divide change in uvw by change in time
        res = uvw_diff / time_diff
        # Assign the result, duplicating the last row's value
        Δuvw_dt[mask] = np.concatenate([res, res[-2:-1, :]])

    return Δuvw_dt
