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


def is_flagged_factory(have_flag_row):
    if have_flag_row:
        def impl(flag_row, r):
            return flag_row[r] != 0
    else:
        def impl(flag_row, r):
            return False

    return njit(nogil=True, cache=True)(impl)


def output_factory(have_flag_row):
    if have_flag_row:
        def impl(rows, flag_row):
            return np.zeros(rows, dtype=flag_row.dtype)
    else:
        def impl(rows, flag_row):
            return None

    return njit(nogil=True, cache=True)(impl)


def set_flag_row_factory(have_flag_row):
    if have_flag_row:
        def impl(flag_row, in_row, out_flag_row, out_row, flagged):
            if flag_row[in_row] == 0 and flagged:
                raise RowMapperError("Unflagged input row contributing "
                                     "to flagged output row. "
                                     "This should never happen!")

            out_flag_row[out_row] = (1 if flagged else 0)
    else:
        def impl(flag_row, in_row, out_flag_row, out_row, flagged):
            pass

    return njit(nogil=True, cache=True)(impl)

RowMapOutput = namedtuple("RowMapOutput",
                          ["map", "time", "flag_row"])

@generated_jit(nopython=True, nogil=True, cache=True)
def baseline_row_mapper(uvw, time, antenna1, antenna2, flag_row=None, bins_for_longest_baseline=None):
    
    have_flag_row = not is_numba_type_none(flag_row)
    is_flagged_fn = is_flagged_factory(have_flag_row)

    output_flag_row = output_factory(have_flag_row)
    set_flag_row = set_flag_row_factory(have_flag_row)
    
    def impl(uvw, time, antenna1, antenna2, flag_row=None, bins_for_longest_baseline=None):
        
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
        bin_flagged = np.zeros((nbl, ntime), dtype=np.bool_)
        
#         for r in range(uvw.shape[0]):
        for r in range(time.shape[0]):
            bl = bl_inv[r]
            t = time_inv[r]
            row_lookup[bl, t] = r
            
        for bl in range(ubl.shape[0]):
            tbin = numba.int32(0)
            bin_count = numba.int32(0)
            bin_flag_count = numba.int32(0)
            print("tbin ", tbin)
            print("bin_count", bin_count)
            print("bin_flag_count", bin_flag_count)
                
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
                        bin_flagged[bl, tbin] = bin_count == bin_flag_count
                    else:
                        time_lookup[bl, tbin] = sentinel
                        bin_flagged[bl, tbin] = False
                        
                    # Reset the bin_count, bin_flag_count
                    # increase tbin counts
                    bin_count = 0
                    bin_flag_count = 0
                    tbin += 1
                    print("tbin ", tbin)
                    print("bin_count", bin_count)

                bin_lookup[bl, t] = tbin
                time_lookup[bl, tbin] += time[r]
                bin_count += 1
                print("bin_count", bin_count)
                print("------")
                
                # Record flags - from row_lookup
                if is_flagged_fn(flag_row, r):
                    bin_flag_count += 1

            if bin_count > 0:
                time_lookup[bl, tbin] /= bin_count
                bin_flagged[bl, tbin] = bin_count == bin_flag_count
                tbin += 1
                print("tbin", tbin)

            out_rows += tbin
            print("out_rows", out_rows)
            print("------------")
        
            for b in range(tbin, ntime):
                time_lookup[bl, b] = sentinel
                bin_flagged[bl, b] = False
         
        
        flat_time = time_lookup.ravel() 
        argsort = np.argsort(flat_time, kind='mergesort')
        
        for i, a in enumerate(argsort):
            inv_argsort[a] = i
        
        print("flat_time\n", flat_time)
        print("argsort\n", argsort)
        print("inv_argsort\n", inv_argsort)
        print("ntime", ntime)
        row_map = np.empty((time.shape[0]), dtype=np.uint32)
        
        out_flag_row = output_flag_row(out_rows, flag_row)
        
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
                
            # Flags output
            set_flag_row(flag_row, in_row, out_flag_row, out_row, bin_flagged[bl, tbin])

            row_map[in_row] = out_row
            
        time_ret = flat_time[argsort[:out_rows]]
        print(row_map)
        print(time_ret)
        
        return RowMapOutput(row_map, time_ret, out_flag_row) 
    
    return impl


@jit(nopython=True, nogil=True, cache=True)
def baseline_channel_mapper():
    pass
