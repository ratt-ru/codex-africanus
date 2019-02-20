# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from africanus.util.docs import DocstringTemplate

import numpy as np
import numba


@numba.njit(nogil=True, cache=True)
def _time_and_chan_avg(time, vis, utime, time_inv, ubl, bl_inv,
                       time_bins, chan_bins):

    if len(vis.shape) != 3:
        raise ValueError("visibilities must have shape (row, chan, corr)")

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

    # Determine averaging across time and channel
    # (same as number of elements in each bin)
    avg_times = time_bin_size = (ntime + time_bins - 1) // time_bins
    avg_chans = chan_bin_size = (nchan + chan_bins - 1) // chan_bins

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

    # print(argsort)
    # print(inv_argsort)

    # Allocate output
    output = np.zeros((out_rows, chan_bins, ncorr), dtype=vis.dtype)

    # print(output.shape, lookup.shape, argsort.shape)

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
                    # chan_count = 0
                    # cbin += 1
                    pass

            valid_times += 1

            # If we've completely filled the time bin
            # normalise by time count
            if valid_times == time_bin_size:
                # for f in range(chan_bins):
                #     for c in range(ncorr):
                #         output[orow, f, c] /= valid_times

                valid_times = 0
                tbin += 1
                pass

        # Normalise any remaining data in the last time bin
        if valid_times > 0:
            # for f in range(chan_bins):
            #     for c in range(ncorr):
            #         output[orow, f, c] /= valid_times

            # valid_times = 0
            # tbin += 1
            pass

    return output


def time_and_channel(time, ant1, ant2, vis, time_bins=1, chan_bins=1):
    utime, time_inv = np.unique(time, return_inverse=True)
    bl = np.stack([ant1, ant2], axis=1)
    ubl, bl_inv = np.unique(bl, axis=0, return_inverse=True)

    return _time_and_chan_avg(time, vis,
                              utime, time_inv, ubl, bl_inv,
                              time_bins, chan_bins)


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

Returns
-------
averaged_visibilities : $(array_type)
    Averaged visibilities of shape :code:`(row, chan, corr)`

""")


try:
    time_and_channel.__doc__ = TIME_AND_CHANNEL_DOCS.substitute(
                                    array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass
