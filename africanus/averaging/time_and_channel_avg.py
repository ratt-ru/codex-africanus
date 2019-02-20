# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from operator import mul

import numpy as np
import numba

from africanus.compatibility import reduce
from africanus.util.docs import DocstringTemplate


@numba.jit(nogil=True, cache=True)
def _time_and_chan_avg(time, ant1, ant2, vis,
                       utime, time_inv, ubl, bl_inv,
                       time_bins, chan_bins,
                       corr_shape,
                       return_time=False,
                       return_antenna=False):

    nbl = ubl.shape[0]
    ntime = utime.shape[0]
    nchan = vis.shape[1]
    ncorr = vis.shape[2]

    # Create a bl x time mask representing the full
    # resolution matrix possible for this chunk of rows
    mask = np.full((nbl, ntime), -1, dtype=np.int32)

    # Fill mask indicating presence of row data
    for r in range(vis.shape[0]):
        ti = time_inv[r]
        bli = bl_inv[r]
        mask[bli, ti] = r

    # Determine bins size for time and channel
    time_bin_size = (ntime + time_bins - 1) // time_bins
    chan_bin_size = (nchan + chan_bins - 1) // chan_bins

    # Create a lookup table of averaged times for each baseline,
    # used to order visibilities in the output data.
    # time_sentinel, indicating absence of data,
    # set to maximum floating point value so that
    # missing data is moved to the end during an argsort of the lookup
    time_sentinel = np.finfo(time.dtype).max
    lookup_shape = (nbl, time_bins)
    lookup = np.full(lookup_shape, time_sentinel, dtype=time.dtype)

    # Number of output rows
    out_rows = 0

    # For each baseline, average associated times
    for bli in range(ubl.shape[0]):
        tbin = numba.int32(0)         # Time averaging bin
        valid_times = numba.int32(0)  # Number of time samples

        for ti in range(utime.shape[0]):
            r = mask[bli, ti]

            # Ignore non-existent entries
            if r == -1:
                continue

            # If we encounter the sentinel value, just assign
            # otherwise add the value to the bin
            if lookup[bli, tbin] == time_sentinel:
                lookup[bli, tbin] = time[r]
            else:
                lookup[bli, tbin] += time[r]

            valid_times += 1

            # If we've completely filled the time averaging bin,
            # normalise it and advance to the next
            if valid_times == time_bin_size:
                lookup[bli, tbin] /= valid_times
                tbin += 1
                valid_times = 0

        # Handle normalisation of the last bin
        if valid_times > 0:
            lookup[bli, tbin] /= valid_times
            tbin += 1

        # Add number of bins to the output rows
        out_rows += tbin

    # Sort the averaged time values to determine their
    # location in the output. We use mergesort so that the
    # sort is stable.
    argsort = np.argsort(lookup.ravel(), kind='mergesort')
    inv_argsort = np.empty_like(argsort)

    for i, a in enumerate(argsort):
        inv_argsort[a] = i

    # Allocate output
    output = np.zeros((out_rows, chan_bins, ncorr), dtype=vis.dtype)

    new_time = np.empty((out_rows,), dtype=time.dtype)
    new_ant1 = np.empty((out_rows,), dtype=ant1.dtype)
    new_ant2 = np.empty((out_rows,), dtype=ant2.dtype)

    # For each baseline, average the visibility data
    # along the channel and time dimensions
    for bli in range(nbl):
        tbin = numba.int32(0)         # Time averaging bin
        valid_times = numba.int32(0)  # Number of time samples
        bl_off = bli*time_bins        # Offset of this baseline in flat lookup
        orow = argsort.dtype.type(0)  # Output row

        for ti in range(ntime):
            r = mask[bli, ti]

            # Ignore missing values
            if r == -1 or lookup[bli, tbin] == time_sentinel:
                continue

            orow = inv_argsort[bl_off + tbin]

            new_time[orow] = lookup[bli, tbin]
            new_ant1[orow] = ant1[r]
            new_ant2[orow] = ant2[r]

            for c in range(ncorr):
                cbin = numba.int32(0)        # Channel averaging bin
                chan_count = numba.int32(0)  # Counter variable

                for f in range(nchan):
                    output[orow, cbin, c] += vis[r, f, c]
                    chan_count += 1

                    # If we've completely filled the channel bin
                    # normalise by channel count
                    if chan_count == chan_bin_size:
                        # output[orow, cbin, c] /= chan_count
                        chan_count = 0
                        cbin += 1

                # Normalise any remaining data in the last channel bin
                if chan_count > 0:
                    # output[orow, cbin, c] /= chan_count
                    chan_count = 0
                    cbin += 1

            valid_times += 1

            # If we've completely filled the time bin
            # normalise by time count
            if valid_times == time_bin_size:
                # for f in range(chan_bins):
                #     for c in range(ncorr):
                #         output[orow, f, c] /= valid_times

                valid_times = 0
                tbin += 1

        # Normalise any remaining data in the last time bin
        if valid_times > 0:
            # for f in range(chan_bins):
            #     for c in range(ncorr):
            #         output[orow, f, c] /= valid_times

            valid_times = 0
            tbin += 1

    return_shape = output.shape[:2] + corr_shape

    return output.reshape(return_shape), new_time, new_ant1, new_ant2

    # if return_time and return_antenna:
    #     return output, new_time, new_ant1, new_ant2
    # elif return_time and not return_antenna:
    #     return output, new_time, None, None
    # elif not return_time and return_antenna:
    #     return output, None, new_ant1, new_ant2
    # else:
    #     return output, None, None, None


def time_and_channel(time, ant1, ant2, vis,
                     time_bins=1, chan_bins=1,
                     return_time=False,
                     return_antenna=False):
    utime, time_inv = np.unique(time, return_inverse=True)
    bl = np.stack([ant1, ant2], axis=1)
    ubl, bl_inv = np.unique(bl, axis=0, return_inverse=True)

    if len(vis.shape) < 3:
        raise ValueError("vis must have shape (row, chan, corr1, ..., corr2)")
    # Flatten correlations if necessary
    elif len(vis) > 3:
        corrs = vis.shape[2:]
        fcorr = reduce(mul, vis.shape[2:], 1)
        vis = vis.reshape(vis.shape[:2] + (fcorr,))
    else:
        corrs = vis.shape[2:]

    result = _time_and_chan_avg(time, ant1, ant2, vis,
                                utime, time_inv, ubl, bl_inv,
                                time_bins, chan_bins,
                                corrs,
                                return_time=return_time,
                                return_antenna=return_antenna)

    if return_time and return_antenna:
        return result
    elif return_time and not return_antenna:
        return result[0], result[1]
    elif not return_time and return_antenna:
        return result[0], result[2], result[3]
    else:
        return result[0]


TIME_AND_CHANNEL_DOCS = DocstringTemplate("""
Average visibility data over time and channel.

Parameters
----------
time : $(array_type)
    time data of shape :code:`(row,)`
antenna1 : $(array_type)
    antenna1 of shape :code:`(row,)`
antenna2 : $(array_type)
    antenna2 of shape :code:`(row,)`
vis : $(array_type)
    visibility data of shape :code:`(row, chan, corr)`
time_bins : int, optional
    Defaults to 1
chan_bins : int, optional
    Defaults to 1
return_time : {True, False}
    Return time centroids of averaged visibilities.
    Defaults to False.
return_antenna : {True, False}
    Return antenna pairs (baseline) of averaged visibilities.
    Defaults to False.

Returns
-------
averaged_visibilities : $(array_type)
    Averaged visibilities of shape :code:`(row, chan, corr)`
averaged_time : $(array_type), optional
    Averaged time centroids of shape :code:`(row,)`
averaged_antenna1 : $(array_type), optional
    antenna1 of averaged visibilities of shape :code:`(row,)`
averaged_antenna2 : $(array_type), optional
    antenna2 of averaged visibilities of shape :code:`(row,)`

""")


try:
    time_and_channel.__doc__ = TIME_AND_CHANNEL_DOCS.substitute(
                                    array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass
