# -*- coding: utf-8 -*-


import numpy as np
import numba

from africanus.util.numba import generated_jit, njit


@njit(nogil=True, cache=True)
def _unique_internal(data):
    if len(data.shape) != 1:
        raise ValueError("_unique_internal currently "
                         "only supports 1D arrays")

    # Handle the empty array case
    if data.shape[0] == 0:
        return (data,
                np.empty((0,), dtype=np.intp),
                np.empty((0,), dtype=np.intp),
                np.empty((0,), dtype=np.intp))

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

    # (uniques, indices, inverse index, counts)
    return aux[mask], perm[mask], inv_idx, np.diff(np.array(counts))


@generated_jit(nopython=True, nogil=True, cache=True)
def unique_time(time):
    """ Return unique time, inverse index and counts """
    if time.dtype not in (numba.float32, numba.float64):
        raise ValueError("time must be floating point but is %s" % time.dtype)

    def impl(time):
        return _unique_internal(time)

    return impl


@generated_jit(nopython=True, nogil=True, cache=True)
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

        ret, idx, inv, counts = _unique_internal(bl)

        # Recast to int32 and reshape
        ubl = ret.view(np.int32).reshape(ret.shape[0], 2)

        return ubl, idx, inv, counts

    return impl
