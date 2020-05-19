# -*- coding: utf-8 -*-

"""
References

Synthesis Imaging in Radio Astronomy II
Lecture 18. Bandwidth and Time-Average Smearing.

DDFacet
Faceting for direction-dependent spectral deconvolution
https://arxiv.org/pdf/1712.02078.pdf
"""

import numpy as np

from africanus.constants import c

# 𝞓 𝝿 𝞇 𝞍 𝝼


def decorrelation_map(time, uvw, ant1, ant2,
                      chan_freq, chan_width,
                      a1, a2, l_max=1,
                      decorrelation=0.02):
    """
    Parameters
    ----------
    time : :class:`numpy.ndarray`
        Shape :code:`(row,)`
    uvw  : :class:`numpy.ndarray`
        Shape :code:`(row, 3)`
    ant1 : :class:`numpy.ndarray`
        Shape :code:`(row,)`
    ant2 : :class:`numpy.ndarray`
        Shape :code:`(row,)`
    chan_freq : :class:`numpy.ndarray`
        Shape :code:`(chan,)`
    chan_width : :class:`numpy.ndarray`
        Shape :code:`(chan,)`
    a1 : int
        First antenna of the baseline
    a2 : int
        Second antenna of the baseline
    l_max : float
        A float between 0.0 and 1.0 indicating the maximum distance
        considered in the lm plane.
    decorrelation : float
        A float between 0.0 and 1.0 indicating the Decorrelation Factor
        or, the reduction in amplitude, considered acceptable during
        the averaging process.
        0.0 implies no decorrelation while 1.0 implies complete decorrelation.
        Defaults to 0.02.
    """

    if l_max < 0.0 or l_max > 1.0:
        raise ValueError("0.0 <= l_max <= 1.0 required %s" % l_max)

    if decorrelation < 0.0 or decorrelation > 1.0:
        raise ValueError("0.0 <= decorrelation <= 1.0 required %s"
                         % decorrelation)

    # NOTE(sjperkins)
    # Symbols used here follow the convention used in
    # Faceting for direction-dependent spectral deconvolution
    #
    # 𝞍 is the phase. Related to Equation 37
    # 𝝼 is the frequency. Related to Equation 38
    # The two are related two each other by the decorrelation factor Equation (36)
    rows = np.logical_and(a1 == ant1, a2 == ant2)
    uvw = uvw[rows]
    time = time[rows]
    nrows = uvw.shape[0]
    nchan = chan_freq.shape[0]

    two_pi_over_c = 2.0 * np.pi / c
    n_max = np.abs(np.sqrt(1.0 - l_max**2) - 1.0)
    𝝼_max = chan_freq.max()

    # Calculate delta UVW
    𝞓uvw = np.zeros_like(uvw)
    𝞓uvw[:-1, :] = uvw[:-1, :] - uvw[1:, :]
    # last delta copied from previous row
    𝞓uvw[-1, :] = 𝞓uvw[-2, :]

    # This looks like the phase rate
    # in Synthesis and Imaging Equation 18-30
    # obtained via the decorrelation factor
    # approximation in Equation 18-31.
    # We're still missing a a frequency factor,
    # also missing from max_𝞍 below where it
    # would presumably be the same (𝝼_max for e.g.)
    𝞓𝞇 = 2 * np.sqrt(6 * decorrelation)

    # max phase \vec{u}\cdot\vec{l} is ||(u,v)||*l_max +|w|*n_max
    # Note missing frequency, also missing from 𝞓𝞍.
    max_𝞍 = (two_pi_over_c *
             (np.sqrt((uvw[:, :2]**2).sum(axis=1))*l_max +
              np.abs(uvw[:, 2])*n_max))
    # Derive max bandwith rate (delta nu) for each row, corresponding to 𝞓𝞍
    # This might correspond to Equation (36) in DDFacet
    𝞓𝝼 = 𝞓𝞇 / max_𝞍

    # Maximum delta phase for each row
    # Occurs when 𝞓uvw lines up with lmn - 1.
    # So assume we have an lmn vector such
    # that ||(l,m)||=l_max, n_max=|sqrt(1 - l_max^2) - 1|
    # the max phase delta for each row will be ||(𝞓u,𝞓v)||*l_max + |𝞓w|*n_max
    # This is Equation (37) in DDFacet with delta nu replaced with 𝝼_max
    max_𝞓𝞍 = ((two_pi_over_c * 𝝼_max) *
              (np.sqrt((𝞓uvw[:, :2]**2).sum(axis=1))*l_max +
               np.abs(𝞓uvw[:, 2])*n_max))

    block_slices = []
    start = 0
    row_𝞓𝞍_sum = 0.0

    # row_𝞓𝞍[i] is distance of row i+1 wrt row i
    for row, (row_𝞓𝞍, row_𝞓𝝼) in enumerate(zip(max_𝞓𝞍, 𝞓𝝼)):
        row_𝞓𝞍_sum += row_𝞓𝞍

        # if more than critical, then block is [start,row + 1)
        if row_𝞓𝞍_sum > 𝞓𝞇:
            block_slices.append(slice(start, row + 1))
            row_𝞓𝞍_sum = 0.0
            start = row + 1

    # add last block
    if start < nrows:
        block_slices.append(slice(start, nrows))

    print(block_slices)

    frac_bandwidth = chan_width / chan_width.sum()

    print(𝞓𝝼)

    # The fractional channel block size.
    # Change in frequency per row divided by channel width per channel
    # frac_chan_block_size = 𝞓𝝼 / chan_width
    frac_chan_block_size = 𝞓𝝼[:, None] / chan_width[None, :]

    # Clamp fractional channel block size to a lower bound of 1 per block
    frac_chan_block_min = np.array([max(frac_chan_block_size[slc].min(), 1)
                                    for slc in block_slices])

    # Convert to integer number of channel blocks for each row
    num_chan_blocks = np.ceil(nchan / frac_chan_block_min)

    # Convert back to integer channel size
    # (smaller than the fractional size, tiling the space
    # more evenly
    size_chan_block = np.int32(np.ceil(nchan / num_chan_blocks))

    # Unique number of channel block sizes
    unique_chan_block_sizes = np.unique(size_chan_block)
    nuniq_blocks = unique_chan_block_sizes.shape[0]

    chan_range = np.arange(nchan, dtype=np.int32)

    chan_pairs = np.zeros((nuniq_blocks, nchan + 1, 2), dtype=np.intp)
    chan_pairs[:, :-1, 0] = chan_range[None,
                                       :] // unique_chan_block_sizes[:, None]
    chan_pairs[:, :-1, 1] = chan_pairs[:, :-1, 0] + 1

    import pdb
    pdb.set_trace()
    print(frac_chan_block_min)
    print(num_chan_blocks)
    print(size_chan_block)
    print(unique_chan_block_sizes)
    print(chan_pairs)


def decorrelation(uvw, 𝞓uvw_𝞓time, interval,
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
        phase = (𝞓uvw_𝞓time[:, 0] * lm[0] +
                 𝞓uvw_𝞓time[:, 1] * lm[1] +
                 𝞓uvw_𝞓time[:, 2] * n) * interval

        phi = np.pi * phase[:, None] * frequency[None, :] / c
        non_zero = phi != 0.0
        phi = phi[non_zero]
        factor[non_zero] *= np.sin(phi)/phi

    return factor


def 𝞓uvw_𝞓time(time, antenna1, antenna2, uvw):
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
    𝞓uvw_dt : :class:`numpy.ndarray`
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

    𝞓uvw_dt = np.zeros_like(uvw)

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
        𝞓uvw_dt[mask] = np.concatenate([res, res[-2:-1, :]])

    return 𝞓uvw_dt
