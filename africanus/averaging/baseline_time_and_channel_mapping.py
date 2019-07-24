# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import numpy as np
import numba

from africanus.averaging.support import unique_time, unique_baselines
from africanus.util.numba import is_numba_type_none, generated_jit, njit, jit

class RowMapperError(Exception):
    pass

RowMapOutput = namedtuple("RowMapOutput",
                          ["map", "time", "interval", "flag_row"])

@generated_jit(nopython=True, nogil=True, cache=True)
def baselibe_row_mapper(uvw, time, antenna1, antenna2, bins_for_longest_baseline=None):
    
    def impl(uvw, time, antenna1, antenna2, bins_for_longest_baseline=None):
        
        bl_uwv_dist = []
        longest_baseline = np.sqrt((uvw**2).sum(axis=1)).max()
        
        na = np.max(antenna2) + 1
        for p in range(na): 
            for q in range(p+1, na): 
                uvwpq = uvw[(antenna1 == p) & (antenna2 == q)] 
                baseline_dist = np.sqrt((uvwpq**2).sum(axis=1))[0]
                bl_uvw_dist.append(baseline_dist)
    
        bl_uvw_dist = np.array(bl_uvw_dist)
        bl_dist_time_bins = longest_baseline//bl_uvw_dist
        
        ubl, bl_idx, bl_inv, bl_count = unique_baselines(antenna1, antenna2)
        utime, _, time_inv, _ = unique_time(time)
        
        
        nbl = ubl.shape[0]
        ntime = utime.shape[0]

        sentinel = np.finfo(time.dtype).max
        out_rows = numba.uint32(0)
        
        scratch = np.full(3*nbl*ntime, -1, dtype=np.int32)
        row_lookup = scratch[:nbl*ntime].reshape(nbl, ntime)
        bin_lookup = scratch[nbl*ntime:2*nbl*ntime].reshape(nbl, ntime)
        time_lookup = np.zeros((nbl, ntime), dtype=time.dtype)
        inv_argsort = scratch[2*nbl*ntime:]
        
        for r in range(uvw.shape[0]):
            bl = bl_inv[r]
            t = time_inv[r]
            row_lookup[bl, t] = r
            
         
        for bl in range(ubl.shape[0]):
            tbin = numba.int32(0)
            bin_count = numba.int32(0)
            bin_flag_count = numba.int32(0)
            bin_low = time.dtype.type(0)

            # range of baseline element count
            for t in range(bl_count[0]):
                # Lookup input row
                r = row_lookup[bl, t]
                # Ignore if not present
                if r == -1:
                    continue

                if bin_count >= bl_dist_time_bins[bl]:
                    # Reset the bin_count
                    # increase tbin counts
                    bin_count = 0
                    tbin += 1

                bin_lookup[bl, t] = tbin
                time_lookup[bl, tbin] += time[r]
                bin_count += 1


            if bin_count > 0:
                time_lookup[bl, tbin] /= bin_count
                tbin += 1

            out_rows += tbin
        
            for b in range(tbin, ntime):
                time_lookup[bl, b] = sentinel
            
        flat_time = time_lookup.ravel() 
        argsort = np.argsort(flat_time, kind='mergesort')
        
        for i, a in enumerate(argsort):
            inv_argsort[a] = i
        
        row_map = np.empty((time.shape[0]), dtype=np.uint32)
        
        for in_row in range(time.shape[0]):
            # Lookup baseline and time
            bl = bl_inv[in_row]
            t = time_inv[in_row]

            # lookup time bin and output row
            tbin = bin_lookup[bl, t]
            # lookup output row in inv_argsort
            out_row = inv_argsort[bl*ntime + tbin]

            if out_row >= out_rows:
                raise RowMapperError("out_row >= out_rows")

            row_map[in_row] = out_row
            time_ret = flat_time[argsort[:out_rows]]
        
        return RowMapOutput(row_map, time_ret) 
    
    return impl


@jit(nopython=True, nogil=True, cache=True)
def baseline_channel_mapper():
    pass
