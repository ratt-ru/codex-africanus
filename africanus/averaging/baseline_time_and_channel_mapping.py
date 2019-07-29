# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import numpy as np
import numba

import sys
sys.path.insert(0, '/Users/smasoka/Varsity/codex-africanus/africanus/averaging/')
from support import unique_time, unique_baselines
from africanus.util.numba import is_numba_type_none, generated_jit, njit, jit

class RowMapperError(Exception):
    pass

RowMapOutput = namedtuple("RowMapOutput",
                          ["map", "time"])

@generated_jit(nopython=True, nogil=True, cache=True)
def baseline_row_mapper(uvw, time, antenna1, antenna2, bins_for_longest_baseline=None):
    
    def impl(uvw, time, antenna1, antenna2, bins_for_longest_baseline=None):
        
        bl_uvw_dist = []
        longest_baseline = np.sqrt((uvw**2).sum(axis=1)).max()
        
        # Unique baseline and time 
        ubl, bl_idx, bl_inv, bl_count = unique_baselines(antenna1, antenna2)
        utime, _, time_inv, _ = unique_time(time)
        
        # Calculate distances of the unique baselines
        for i in range(ubl.shape[0]): 
            p,q = ubl[i,0],ubl[i,1]
            uvwpq = uvw[(antenna1 == p) & (antenna2 == q)] 
            baseline_dist = np.sqrt((uvwpq**2).sum(axis=1))[0]
            bl_uvw_dist.append(baseline_dist)
    
        bl_uvw_dist = np.array(bl_uvw_dist)
        bl_dist_time_bins = longest_baseline//bl_uvw_dist
        print("longest_baseline", longest_baseline)
        print("bl_uvw_dist", bl_uvw_dist)
        print("bl_dist_time_bins", bl_dist_time_bins)
        
        print("ubl \n", ubl)
        
        nbl = ubl.shape[0]
        ntime = utime.shape[0]
        print("nbl", nbl, "ntime", ntime)
        sentinel = np.finfo(time.dtype).max
        out_rows = numba.uint32(0)
        
        scratch = np.full(3*nbl*ntime, -1, dtype=np.int32)
        row_lookup = scratch[:nbl*ntime].reshape(nbl, ntime)
        bin_lookup = scratch[nbl*ntime:2*nbl*ntime].reshape(nbl, ntime)
        time_lookup = np.zeros((nbl, ntime), dtype=time.dtype)
        inv_argsort = scratch[2*nbl*ntime:]
        
#         for r in range(uvw.shape[0]):
        for r in range(time.shape[0]):
            bl = bl_inv[r]
            t = time_inv[r]
            row_lookup[bl, t] = r
            
        for bl in range(ubl.shape[0]):
            tbin = numba.int32(0)
            bin_count = numba.int32(0)
            bin_flag_count = numba.int32(0)
            bin_low = time.dtype.type(0)
            print("tbin ", tbin)
            print("bin_count", bin_count)
            print("bin_flag_count", bin_flag_count)
            print("bin_low", bin_low)
                
            # range of baseline element count
#             for t in range(bl_count[0]):
            for t in range(utime.shape[0]):
                # Lookup input row
                r = row_lookup[bl, t]
                # Ignore if not present
                if r == -1:
                    continue

                if bin_count >= bl_dist_time_bins[bl]:
                    # Normalise
                    if bin_count > 0:
                        time_lookup[bl, tbin] /= bin_count
                    else:
                        time_lookup[bl, tbin] = sentinel
                        
                    # Reset the bin_count
                    # increase tbin counts
                    bin_count = 0
                    tbin += 1
                    print("tbin ", tbin)
                    print("bin_count", bin_count)

                bin_lookup[bl, t] = tbin
                time_lookup[bl, tbin] += time[r]
                bin_count += 1
                print("bin_count", bin_count)
                print("------")


            if bin_count > 0:
                time_lookup[bl, tbin] /= bin_count
                tbin += 1
                print("tbin", tbin)

            out_rows += tbin
            print("out_rows", out_rows)
            print("------------")
        
            for b in range(tbin, ntime):
                time_lookup[bl, b] = sentinel
         
        
        flat_time = time_lookup.ravel() 
        argsort = np.argsort(flat_time, kind='mergesort')
        
        for i, a in enumerate(argsort):
            inv_argsort[a] = i
        
        print("flat_time\n", flat_time)
        print("argsort\n", argsort)
        print("inv_argsort\n", inv_argsort)
        print("ntime", ntime)
        row_map = np.empty((time.shape[0]), dtype=np.uint32)
        
        print("out_rows", out_rows)
        for in_row in range(time.shape[0]):
            # Lookup baseline and time
            bl = bl_inv[in_row]
            t = time_inv[in_row]
            print("bl", bl)
            print("t", t)
            # lookup time bin and output row
            tbin = bin_lookup[bl, t]
            print("bin_lookup\n", bin_lookup)
            print("tbin", tbin)
            # lookup output row in inv_argsort
            out_row = inv_argsort[bl*ntime + tbin]
            
            print("in_row", in_row)
            print("out_row", out_row)
            print("out_rows", out_rows)
            if out_row >= out_rows:
                raise RowMapperError("out_row >= out_rows")

            row_map[in_row] = out_row
            
        time_ret = flat_time[argsort[:out_rows]]
        print(row_map)
        print(time_ret)
        
        return RowMapOutput(row_map, time_ret) 
    
    return impl


@jit(nopython=True, nogil=True, cache=True)
def baseline_channel_mapper():
    pass
