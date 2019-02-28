# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numba

from africanus.util.numba import is_numba_type_none


@numba.njit(nogil=True, cache=True)
def _unique_internal_inverse(data):
    if len(data.shape) != 1:
        raise ValueError("_unique_internal_inverse currently "
                         "only supports 1D arrays")

    # See numpy's unique1d
    perm = np.argsort(data, kind='mergesort')

    # Combine these arrays to save on allocations?
    aux = np.empty_like(data)
    mask = np.empty(aux.shape, dtype=np.bool_)
    inv_idx = np.empty(mask.shape, dtype=np.intp)

    # Hard code first iteration
    p = perm[0]
    aux[0] = data[p]
    mask[0] = True
    cumsum = 1
    inv_idx[p] = cumsum - 1
    counts = [np.intp(0)]

    for i in range(1, aux.shape[0]):
        p = perm[i]
        aux[i] = data[p]
        d = aux[i] != aux[i - 1]
        mask[i] = d
        cumsum += d
        inv_idx[p] = cumsum - 1

        if d:
            counts.append(np.intp(i))

    counts.append(aux.shape[0])

    return aux[mask], inv_idx, np.diff(np.array(counts))


@numba.generated_jit(nopython=True, nogil=True, cache=True)
def unique_time(time):
    """ Return unique time, inverse index and counts """
    if time.dtype not in (numba.float32, numba.float64):
        raise ValueError("time must be floating point but is %s" % time.dtype)

    def impl(time):
        return _unique_internal_inverse(time)

    return impl


@numba.generated_jit(nopython=True, nogil=True, cache=True)
def unique_baselines(ant1, ant2):
    """ Return unique baselines, inverse index and counts """
    if not ant1.dtype == numba.int32 or not ant2.dtype == numba.int32:
        # Need these to be int32 for the bl_32bit.view(np.int64) trick
        raise ValueError("ant1 and ant2 must be np.int32 "
                         "but received %s and %s" %
                         (ant1.dtype, ant2.dtype))

    def impl(ant1, ant2):
        # Trickery, stack the two int32 antenna pairs in an array
        # and cast to int64
        bl_32bit = np.empty((ant1.shape[0], 2), dtype=np.int32)

        # Copy data
        for r in range(ant1.shape[0]):
            bl_32bit[r, 0] = ant1[r]
            bl_32bit[r, 1] = ant2[r]

        # Cast to int64 for the unique operation
        bl = bl_32bit.view(np.int64).reshape(ant1.shape[0])

        ret, inv, counts = _unique_internal_inverse(bl)

        # Recast to int32 and reshape
        ubl = ret.view(np.int32).reshape(ret.shape[0], 2)

        return ubl, inv, counts

    return impl


def row_or_minus_one_factory(flag_row, flag):
    """
    Factory function returning a function which returns
    -1 if all values related to a row are flagged,
    or the row itself.

    Parameters
    ----------
    flag_row : :class:`numba.types.npytypes.Array` or \
        :class:`numba.types.misc.Omitted`
    flag : :class:`numba.types.npytypes.Array` or \
        :class:`numba.types.misc.Omitted`

    Returns
    -------
    callable
        Function `f(flag_row, flag, r)` that returns -1
        if everything in the row is flagged and `r` otherwise.
    """

    have_flag_row = not is_numba_type_none(flag_row)
    have_flag = not is_numba_type_none(flag)

    if have_flag and have_flag_row:
        def impl(flag_row, flag, r):
            # Entire row is flagged, we can exit early
            if flag_row[r] != 0:
                return -1

            # Return the row if anything is unflagged
            for f in range(flag.shape[1]):
                for c in range(flag.shape[2]):
                    if flag[r, f, c] == 0:
                        return r

            return -1

    elif have_flag and not have_flag_row:
        def impl(flag_row, flag, r):
            for f in range(flag.shape[1]):
                for c in range(flag.shape[2]):
                    if flag[r, f, c] == 0:
                        return r

            return -1
    elif not have_flag and have_flag_row:
        def impl(flag_row, flag, r):
            return r if flag_row[r] == 0 else -1
    else:
        def impl(flag_row, flag, r):
            return r

    return numba.njit(nogil=True, cache=True)(impl)


@numba.generated_jit(nopython=True, nogil=True, cache=True)
def generate_metadata(time, ant1, ant2, time_bin_size=1,
                      flag_row=None, flag=None):

    row_or_minus_one = row_or_minus_one_factory(flag_row, flag)

    def impl(time, ant1, ant2, time_bin_size=1,
             flag_row=None, flag=None):

        ubl, bl_inv, bl_counts = unique_baselines(ant1, ant2)
        utime, time_inv, time_counts = unique_time(time)

        nbl = ubl.shape[0]
        ntime = utime.shape[0]
        tbins = (ntime + time_bin_size - 1) // time_bin_size

        sentinel = np.finfo(time.dtype).max
        in_lookup = np.full((nbl, ntime), -1, np.int32)
        time_lookup = np.full((nbl, tbins), sentinel, time.dtype)

        out_rows = 0

        for r in range(time.shape[0]):
            ti = time_inv[r]
            bli = bl_inv[r]

            in_lookup[bli, ti] = row_or_minus_one(flag_row, flag, r)

        # For each baseline, average associated times
        for bli in range(in_lookup.shape[0]):
            tbin = numba.int32(0)         # Time averaging bin
            valid_times = numba.int32(0)  # Number of time samples

            for ti in range(in_lookup.shape[1]):
                r = in_lookup[bli, ti]

                # Ignore non-existent entries
                if r == -1:
                    continue

                # If we encounter the sentinel value, just assign
                # otherwise add the value to the bin
                if time_lookup[bli, tbin] == sentinel:
                    time_lookup[bli, tbin] = time[r]
                else:
                    time_lookup[bli, tbin] += time[r]

                valid_times += 1

                # If we've completely filled the time averaging bin,
                # normalise it and advance to the next
                if valid_times == time_bin_size:
                    time_lookup[bli, tbin] /= valid_times
                    tbin += 1
                    valid_times = 0

            # Handle normalisation of the last bin
            if valid_times > 0:
                time_lookup[bli, tbin] /= valid_times
                tbin += 1

            # Add number of bins to the output rows
            out_rows += tbin

        # Sort the averaged time values to determine their
        # location in the output. We use mergesort so that the
        # sort is stable.
        argsort = np.argsort(time_lookup.ravel(), kind='mergesort')
        inv_argsort = np.empty_like(argsort)

        for i, a in enumerate(argsort):
            inv_argsort[a] = i

        return (in_lookup, time_lookup, inv_argsort,
                out_rows, time_bin_size, sentinel)

    return impl


@numba.jit(nopython=True, nogil=True, cache=True)
def better_lookup(time, antenna1, antenna2, time_bin_size=1):
    """
    Generates a mapping from a high resolution row index to
    a low resolution row index in support of time and channel
    averaging code. The `time` column is also averaged during this
    process in order to produce the mapping.

    In order to average a chunk of row data, it is necessary to
    group each row (or sample) by baseline and then average
    the time samples present in each baseline in bins of
    `time_bin_size`. The algorithm is robust in the presence
    of missing time and baseline data.

    The algorithm works as follows:

    1. `antenna1`, `antenna2` and `time`are used to construct a
    `row_lookup` array of shape `(ubl, utime)` mapping a baseline and time
    to a row of input data.

    2. For each baseline, `time_bin_size` times are averaged together
    into a `time_lookup` array of shape `(ubl, tbins)`.
    Not all bins may be filled for a baseline if data is flagged or missing --
    these bins are assigned a sentinel value set to the
    maximum floating point value.
    A secondary `bin_lookup` array of shape `(ubl, utime)` is constructed
    mapping a time in the `row_lookup` array to a time bin in `time_lookup`.

    3. The `time_lookup` array is flattened and argsorted with a stable
    merge sort. As missing values are set to the maximum floating point
    value, this moves valid data to the front and missing data to the back.
    This has the effect of lexicographically sorts the data
    in an ascending `(time, bl)` order

    4. Input rows are then mapped via the `row_lookup`, `bin_lookup`
    and argsorted `time_lookup` arrays to an output row.

    .. code-block:: python

        row_map, time_avg = better_lookup(time, ant1, ant2, time_bin_size=3)

        # Recompute time average using row map
        new_time_avg = np.zeros_like(time_avg)
        ant1_avg = np.zeros(time_avg.shape, dtype=np.int32)
        ant2_avg = np.zeros(time_avg.shape, dtype=np.int32)
        counts = np.empty(time_avg.shape, np.uint32)

        for r in range(time.shape[0]):
            out_row = row_map[r]              # Lookup output row
            count[out_row] += 1               # Advance output row sample

            new_time_avg[out_row] += time[r]  # Sum time values

            # We assign baselines because each input baseline
            # is mapped to the same output baseline
            ant1_avg[out_row] = ant1[r]
            ant2_avg[out_row] = ant2[r]

        # Normalise
        new_time_avg /= count

        np.testing.assert_array_equal(time_avg, new_time_avg)

    Parameters
    ----------
    time : :class:`numpy.ndarray`
        Time values of shape :code:`(row,)`
    antenna1 : :class:`numpy.ndarray`
        Antenna 1 values of shape :code:`(row,)`
    antenna2 : :class:`numpy.ndarray`
        Antenna 2 values of shape :code:`(row,)`
    time_bin_size : int, optional
        Number of timesteps to average into each bin

    Returns
    -------
    row_lookup : :class:`numpy.ndarray`
        Mapping from `np.arange(row)` to output row indices
    time_avg : :class:`numpy.ndarray`
        Averaged time values of shape :code:`(out_row,)`
    """
    ubl, bl_inv, bl_counts = unique_baselines(antenna1, antenna2)
    utime, time_inv, time_counts = unique_time(time)

    nbl = ubl.shape[0]
    ntime = utime.shape[0]
    tbins = (ntime + time_bin_size - 1) // time_bin_size

    sentinel = np.finfo(time.dtype).max
    out_rows = 0

    scratch = np.empty(2*nbl*ntime + nbl*tbins, dtype=np.intp)
    row_lookup = scratch[:nbl*ntime].reshape(nbl, ntime)
    bin_lookup = scratch[nbl*ntime:2*nbl*ntime].reshape(nbl, ntime)
    inv_argsort = scratch[2*nbl*ntime:]
    time_lookup = np.full((nbl, tbins), sentinel, dtype=time.dtype)

    # Construct the row_lookup matrix
    row_lookup[:, :] = -1

    for r in range(time.shape[0]):
        bl = bl_inv[r]
        t = time_inv[r]
        row_lookup[bl, t] = r

    # Average times over each baseline and construct the bin_lookup matrix
    for bl in range(ubl.shape[0]):
        tbin = 0
        bin_contents = 0

        for t in range(utime.shape[0]):
            r = row_lookup[bl, t]

            if r == -1:
                continue

            if time_lookup[bl, tbin] == sentinel:
                time_lookup[bl, tbin] = time[r]
            else:
                time_lookup[bl, tbin] += time[r]

            bin_lookup[bl, t] = tbin
            bin_contents += 1

            if bin_contents == time_bin_size:
                time_lookup[bl, tbin] /= bin_contents
                bin_contents = 0
                tbin += 1

        if bin_contents > 0:
            time_lookup[bl, tbin] /= bin_contents
            tbin += 1

        out_rows += tbin

    # Flatten the time lookup and argsort it
    flat_time = time_lookup.ravel()
    argsort = np.argsort(flat_time, kind='mergesort')

    for i, a in enumerate(argsort):
        inv_argsort[a] = i

    row_map = np.empty(time.shape[0], dtype=np.intp)

    for in_row in range(time.shape[0]):
        bl = bl_inv[in_row]
        t = time_inv[in_row]

        tbin = bin_lookup[bl, t]
        row_map[in_row] = inv_argsort[bl*tbins + tbin]

    return row_map, flat_time[argsort[:out_rows]]
