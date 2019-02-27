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

    for i in range(1, aux.shape[0]):
        p = perm[i]
        aux[i] = data[p]
        d = aux[i] != aux[i - 1]
        mask[i] = d
        cumsum += d
        inv_idx[p] = cumsum - 1

    return aux[mask], inv_idx


@numba.generated_jit(nogil=True, cache=True)
def unique_time(time):
    """ Return unique time and inverse index """
    if time.dtype not in (numba.float32, numba.float64):
        raise ValueError("time must be floating point but is %s" % time.dtype)

    def impl(time):
        return _unique_internal_inverse(time)

    return impl


@numba.generated_jit(nogil=True, cache=True)
def unique_baselines(ant1, ant2):
    """ Return unique baselines and inverse index"""
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

        bl = bl_32bit.view(np.int64).reshape(ant1.shape[0])

        ret, inv = _unique_internal_inverse(bl)

        # Now reshape
        return ret.view(np.int32).reshape(ret.shape[0], 2), inv

    return impl


def flagged_factory(flag_row, flag):
    have_flag_row = not is_numba_type_none(flag_row)
    have_flag = not is_numba_type_none(flag)

    if have_flag and have_flag_row:
        def _minus_one_if_all_flagged(flag_row, flags, r):
            if flag_row[r] == 0:
                return r

            for f in range(flags.shape[1]):
                for c in range(flags.shape[2]):
                    if flags[r, f, c] == 0:
                        return r
            return -1

    elif have_flag and not have_flag_row:
        def _minus_one_if_all_flagged(flag_row, flags, r):
            for f in range(flags.shape[1]):
                for c in range(flags.shape[2]):
                    if flags[r, f, c] == 0:
                        return r

            return -1
    elif not have_flag and have_flag_row:
        def _minus_one_if_all_flagged(flag_row, flags, r):
            return r if flag_row[r] == 0 else -1
    else:
        def _minus_one_if_all_flagged(flag_row, flags, r):
            return r

    return numba.njit(nogil=True, cache=True)(_minus_one_if_all_flagged)


@numba.generated_jit(nopython=True, nogil=True, cache=True)
def generate_lookups(time, ant1, ant2, time_bin_size=1,
                     flag_row=None, flag=None):

    row_or_minus_one = flagged_factory(flag_row, flag)

    def impl(time, ant1, ant2, time_bin_size=1,
             flag_row=None, flag=None):

        ubl, bl_inv = unique_baselines(ant1, ant2)
        utime, time_inv = unique_time(time)

        nbl = ubl.shape[0]
        ntime = utime.shape[0]
        tbins = (ntime + time_bin_size - 1) // time_bin_size

        sentinel = np.finfo(time.dtype).max
        in_lookup = np.full((nbl, ntime), -1, np.int32)
        out_lookup = np.full((nbl, tbins), sentinel, time.dtype)

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
                if out_lookup[bli, tbin] == sentinel:
                    out_lookup[bli, tbin] = time[r]
                else:
                    out_lookup[bli, tbin] += time[r]

                valid_times += 1

                # If we've completely filled the time averaging bin,
                # normalise it and advance to the next
                if valid_times == time_bin_size:
                    out_lookup[bli, tbin] /= valid_times
                    tbin += 1
                    valid_times = 0

            # Handle normalisation of the last bin
            if valid_times > 0:
                out_lookup[bli, tbin] /= valid_times
                tbin += 1

            # Add number of bins to the output rows
            out_rows += tbin

        # Sort the averaged time values to determine their
        # location in the output. We use mergesort so that the
        # sort is stable.
        argsort = np.argsort(out_lookup.ravel(), kind='mergesort')
        inv_argsort = np.empty_like(argsort)

        for i, a in enumerate(argsort):
            inv_argsort[a] = i

        return in_lookup, inv_argsort

    return impl
