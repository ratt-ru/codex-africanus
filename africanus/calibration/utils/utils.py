# -*- coding: utf-8 -*-

import numpy as np
from africanus.util.docs import DocstringTemplate

DIAG_DIAG = 0
DIAG = 1
FULL = 2


def check_type(jones, vis, vis_type='vis'):
    if vis_type == 'vis':
        vis_ndim = (3, 4)
    elif vis_type == 'model':
        vis_ndim = (4, 5)
    else:
        raise ValueError("Unknown vis_type")

    vis_axes_count = vis.ndim
    jones_axes_count = jones.ndim
    if vis_axes_count == vis_ndim[0]:
        mode = DIAG_DIAG
        if jones_axes_count != 5:
            raise RuntimeError("Jones axes not compatible with \
                                visibility axes. Expected length \
                                5 but got length %d" % jones_axes_count)

    elif vis_axes_count == vis_ndim[1]:
        if jones_axes_count == 5:
            mode = DIAG

        elif jones_axes_count == 6:
            mode = FULL
        else:
            raise RuntimeError("Jones term has incorrect shape")
    else:
        raise RuntimeError("Visibility data has incorrect shape")

    return mode


def chunkify_rows(time, utimes_per_chunk):
    utimes, time_bin_counts = np.unique(time, return_counts=True)
    n_time = len(utimes)
    row_chunks = [np.sum(time_bin_counts[i:i+utimes_per_chunk])
                  for i in range(0, n_time, utimes_per_chunk)]
    time_bin_indices = np.zeros(n_time, dtype=np.int32)
    time_bin_indices[1::] = np.cumsum(time_bin_counts)[0:-1]
    time_bin_indices = time_bin_indices.astype(np.int32)
    time_bin_counts = time_bin_counts.astype(np.int32)
    return tuple(row_chunks), time_bin_indices, time_bin_counts


CHECK_TYPE_DOCS = DocstringTemplate("""
    Determines which calibration scenario to apply i.e.
    DIAG_DIAG, DIAG or COMPLEX2x2.

    Parameters
    ----------
    jones : $(array_type)
        Jones term of shape :code:`(time, ant, chan, dir, corr)`
        or :code:`(time, ant, chan, dir, corr, corr)`
    vis : $(array_type)
        Visibility data of shape :code:`(row, chan, corr)`
        or :code:`(row, chan, corr, corr)`
    vis_type : str
        String specifying what kind of visibility we are checking
        against. Options are 'vis' or 'model'

    Returns
    -------
    mode : integer
        An integer representing the calibration mode.
        Options are 0 -> DIAG_DIAG, 1 -> DIAG, 2 -> FULL

""")

try:
    check_type.__doc__ = CHECK_TYPE_DOCS.substitute(
                                    array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass

CHUNKIFY_ROWS_DOCS = DocstringTemplate("""
    Divides rows into chunks containing integer
    numbers of times keeping track of the indices
    at which the unique time changes and the number
    of times per unique time.

    Parameters:
    -----------

    time : $(array_type)
        TIME column of MS
    utimes_per_chunk : integer
        The number of unique times to place in each chunk

    Returns
    -------
    row_chunks : tuple
        A tuple of row chunks that can be used to initialise
        an xds with chunks={'row': row_chunks} for example.
    time_bin_indices : $(array_type)
        Array containing the indices at which unique time
        changes
    times_bin_counts : $(array_type)
        Array containing the number of times per unique time.
""")

try:
    chunkify_rows.__doc__ = CHUNKIFY_ROWS_DOCS.substitute(
                                    array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass
