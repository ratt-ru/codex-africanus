# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from operator import mul

import numba
import numpy as np

from africanus.compatibility import reduce
from africanus.util.docs import DocstringTemplate
from africanus.util.numba import jit


@jit(nopython=True, nogil=True, cache=True)
def _minus_one_if_all_flagged(flags, r):
    for f in range(flags.shape[1]):
        for c in range(flags.shape[2]):
            if flags[r, f, c] == 0:
                return r

    return -1


@jit(nopython=True, nogil=True, cache=True)
def _time_and_chan_avg(time, ant1, ant2, vis, flags,
                       utime, time_inv, ubl, bl_inv,
                       avg_time, avg_chan,
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

        # Indicate absence if all data is flagged for this row
        r = _minus_one_if_all_flagged(flags, r)

        mask[bli, ti] = r

    # Determine bins size for time and channel.
    # using a single bin for each sample if no averaging is indicated
    time_bin_size = 1 if avg_time is None else avg_time
    chan_bin_size = 1 if avg_chan is None else avg_chan

    time_bins = (ntime + time_bin_size - 1) // time_bin_size
    chan_bins = (nchan + chan_bin_size - 1) // chan_bin_size

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

    # Allocate output and scratch space
    output = np.zeros((out_rows, chan_bins, ncorr), dtype=vis.dtype)
    scratch = np.empty((chan_bins, ncorr), dtype=vis.dtype)
    chan_counts = np.empty(ncorr, dtype=np.int32)
    cbins = np.empty(ncorr, dtype=np.int32)

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

            # Lookup output row
            orow = inv_argsort[bl_off + tbin]

            new_time[orow] = lookup[bli, tbin]
            new_ant1[orow] = ant1[r]
            new_ant2[orow] = ant2[r]

            # --------------------
            # Average over channel
            # --------------------

            # Zero scratch space
            for f in range(chan_bins):
                for c in range(ncorr):
                    scratch[f, c] = 0.0

            # Zero per-correlation channel counts and current bin
            for c in range(ncorr):
                chan_counts[c] = 0
                cbins[c] = 0

            # Add any samples into the scratch space
            for f in range(nchan):
                for c in range(ncorr):
                    if flags[r, f, c] != 0:
                        continue

                    scratch[cbins[c], c] += vis[r, f, c]
                    chan_counts[c] += 1

                    # If we've completely filled the channel bin
                    # normalise by the channel count
                    if chan_counts[c] == chan_bin_size:
                        scratch[cbins[c], c] /= chan_counts[c]
                        chan_counts[c] = 0
                        cbins[c] += 1

            # Normalise any remaining data in the last channel bin
            for c in range(ncorr):
                if chan_counts[c] > 0:
                    scratch[cbins[c], c] /= chan_counts[c]
                    chan_counts[c] = 0
                    cbins[c] += 1

            # Copy from the scratch into the output
            for f in range(chan_bins):
                for c in range(ncorr):
                    output[orow, f, c] += scratch[f, c]

            # -----------------
            # Average over time
            # -----------------

            valid_times += 1

            # If we've completely filled the time bin
            # normalise by time count
            if valid_times == time_bin_size:
                for f in range(chan_bins):
                    for c in range(ncorr):
                        output[orow, f, c] /= valid_times

                valid_times = 0
                tbin += 1

        # Normalise any remaining data in the last time bin
        if valid_times > 0:
            for f in range(chan_bins):
                for c in range(ncorr):
                    output[orow, f, c] /= valid_times

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


def time_and_channel(time, ant1, ant2, vis, flags,
                     avg_time=None, avg_chan=None,
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
        flat_vis = vis.reshape(vis.shape[:2] + (fcorr,))
        flat_flags = flags.reshape(flags.shape[:2] + (fcorr,))
    else:
        corrs = vis.shape[2:]
        flat_vis = vis
        flat_flags = flags

    # No averaging requested
    if avg_time is None and avg_chan is None:
        result = vis, time, ant1, ant2
    else:
        result = _time_and_chan_avg(time, ant1, ant2, flat_vis, flat_flags,
                                    utime, time_inv, ubl, bl_inv,
                                    avg_time, avg_chan,
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
    time data of shape :code:`(row,)`.
antenna1 : $(array_type)
    antenna1 of shape :code:`(row,)`.
antenna2 : $(array_type)
    antenna2 of shape :code:`(row,)`.
vis : $(array_type)
    visibility data of shape :code:`(row, chan, corr)`.
flags : $(array_type)
    flags of shape :code:`(row, chan, corr)`.
avg_time : None or int, optional
    Number of times to average into each time bin.
    Defaults to None, in which case no averaging is performed.
avg_chan : None or int, optional
    Number of channels to average into each channel bin.
    Defaults to None, in which case no averaging is performed.
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
