#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import optparse
import numpy as np
import numba
#import bd_averager

from africanus.averaging.support import unique_time, unique_baselines, unique_uvw
from africanus.averaging.time_and_channel_avg import time_and_channel
from africanus.averaging.time_and_channel_mapping import (row_mapper,
                                                          channel_mapper)
from africanus.util.numba import is_numba_type_none, generated_jit, njit, jit

nr_freq = 10 # Number of frequencies
nr_corr = 2  # Number of Correlators
nr_time = 10 # number of times to step
chan_bin_size = 1 # [1, 3, 5]
time_bin_secs = 1 # [1, 2, 3, 4]
year = 2011
month = 11
day = 16
start_time = "17:00:00"
step_time = 2.5 # seconds


#File with Antenna positions
antennasfile = '../bda-dist-pilot/bda_dist_pilot/test_case/meerkat.itrf.txt' 

# Read file into numpy array 
#antennas_pos = np.genfromtxt(antennasfile)[:4,:3]
#antennas_pos = np.genfromtxt(antennasfile)[:4,:3]
antennas_pos = np.genfromtxt(antennasfile)[-3:,:3]
print(antennas_pos)
print(antennas_pos.shape)
print(type(antennas_pos))
print(len(antennas_pos))
na = len(antennas_pos)

# Number of baselines
nant = len(antennas_pos)
bl = []
for i in range(nant):
    for j in range(i+1,nant):
        print(antennas_pos[i], antennas_pos[j])
        bl.append(antennas_pos[i] - antennas_pos[j])

# Need some numpy array functionality
bl = np.array(bl)
print('bl things')
print(bl.shape)
print(bl)
nbr_bl = bl.shape[0]

# convert hh:mm:ss to days
def timetodays(hhmmss):
    days = 0
    hours, mins, sec = map(int, hhmmss.split(':'))
    days = sec + mins + (days / 60.) + hours + (days / 60.)
    
    return days / 24.

print('Time in Days')
print(timetodays(start_time))
days = timetodays(start_time)


# Modified Julian Dates 
jd = (367*year - int(7*(year + int((month+9)/12))/4) - int((3*(int(year + (month - 9)/7)/100)+1)/4) + int(275*month/9) + days + 1721028.5)
mjd = jd - 2400000.5
print('Dates')
print('jd = ',jd)
print('mjd = ',mjd)

# These timestamps should be in Mean Julian Date seconds
# time = np.linspace(0.1, 0.9, nr_time)
mjd *= 86400 
timestamps = np.linspace(mjd, mjd + (step_time * nr_time), num=nr_time)
print('Time', timestamps)

# Get the number of antennas
nbr_ants = antennas_pos.shape[0]

# Tricky things happening here
ant1, ant2 = np.triu_indices(nbr_ants, 1)
ant1 = np.tile(ant1, nr_time)
ant2 = np.tile(ant2, nr_time)
ant1 = np.asarray(ant1, dtype=np.int32)
ant2 = np.asarray(ant2, dtype=np.int32)
print("ant1\n", ant1.dtype)
print("ant2\n", ant2.dtype)

#nr_bl = ant1.size
#print("nr_bl = ",nr_bl)

# Expand these columns up to row = nr_time * nr_bl
timestamps = np.repeat(timestamps, bl.shape[0])
time_centroid = timestamps
scan = np.zeros_like(timestamps, dtype=np.int32)
desc = np.zeros_like(timestamps, dtype=np.int32)
interval = np.repeat(step_time, bl.shape[0] * nr_time)
exposure = interval
weights = np.tile([1,1], (ant1.size,1))
sigma = weights
flag_row = np.repeat(False, ant1.size)

shape = (timestamps.shape[0], nr_freq, nr_corr)
weight_spectrum = np.arange(np.product(shape), dtype=np.float64).reshape(shape)
sigma_spectrum = np.arange(np.product(shape), dtype=np.float64).reshape(shape)

# Flag and Data
# Shape Nn, Nf=10, Nc=2
flag = np.tile([False, False], (nr_freq, 1))
flag = np.repeat(flag[np.newaxis,:,:], ant1.size, axis=0)
vis_data = np.ones_like(flag)


# Random  Hour Angle in Radians
ha = 0.5236
# Random Declination in Radians
# -45:00:00
# 45 * np.pi / 180
dec = -0.7854

# The Big Formula format
ux = [np.sin(ha), np.cos(ha), 0]
vx = [-np.sin(dec)*np.cos(ha), np.sin(dec)*np.sin(ha), np.cos(dec)]
wx = [np.cos(dec)*np.cos(ha), -np.cos(dec)*np.sin(ha), np.sin(dec)]

# construct the U, V, W coordinates from the baselines
u = np.outer(ux[0],bl[:,0]) + np.outer(ux[1],bl[:,1]) + np.outer(ux[2],bl[:,2])
v = np.outer(vx[0],bl[:,0]) + np.outer(vx[1],bl[:,1]) + np.outer(ux[2],bl[:,2])
w = np.outer(wx[0],bl[:,0]) + np.outer(wx[1],bl[:,1]) + np.outer(wx[2],bl[:,2])

u,v,w = [ coord.flatten() for coord in (u,v,w) ]
uvw = np.column_stack((u,v,w))
print(uvw)
uvw = np.tile(uvw, (nr_time, 1))
#print('UVW things')
#print(uvw.shape)
print(uvw[:])


#################################
## The shape of all the arrays ##
################################

print('Number of Antennas')
print(na)
print('Number of Baselines')
print(nbr_bl)
print('Desc')
print(desc.shape)
print('UVW')
print(uvw.shape)
print('Flag')
print(flag.shape)
print('Flag Row')
print(flag_row.shape)
print('Antenna 1')
print(ant1.shape)
print('Antenna 2')
print(ant2.shape)
print('Interval')
print(interval.shape)
print("Exposure")
print(exposure.shape)
print('Time')
print(timestamps.shape)
print(timestamps)
print('Time Centroid')
print(time_centroid.shape)
print('DATA')
print(vis_data.shape)
print('Weights')
print(weights.shape)
print('Scan')
print(scan.shape)

print('------------------\n------------------')

print("Call Time and averaging from africanus\n")

flagged_rows = []
flag_row[flagged_rows] = 1
flag[flagged_rows, :, :] = 1
print('Flag')
print(flag.shape)
print('Flag Row')
print(flag_row.shape)

bl_uvw_dist = []

# From the formula
longest_baseline = np.sqrt((uvw**2).sum(axis=1)).max()

print("\n")
na = np.max(ant2) + 1
print(na)
for p in range(na): 
    for q in range(p+1, na): 
        uvwpq = uvw[(ant1 == p) & (ant2 == q)] 
        print("\n uvwpq\n", uvwpq) 
#        baseline_dist = np.abs(uvwpq[:, 0]).sum(axis=0) 
        baseline_dist = np.sqrt((uvwpq**2).sum(axis=1))[0]
        bl_uvw_dist.append(baseline_dist)
        print("\n baseline_dist\n", baseline_dist)
print("\n")
bl_uvw_dist = np.array(bl_uvw_dist)
bl_dist_time_bins = longest_baseline//bl_uvw_dist

print("bl_uvw_dist \n", bl_uvw_dist, bl_uvw_dist.shape)
print("longest_baseline", longest_baseline)
print("bl_uvw_dist_ratio \n", bl_dist_time_bins, bl_dist_time_bins.shape)


#time_bin_secs = 6
ubl, bl_idx, bl_inv, bl_count = unique_baselines(ant1, ant2)
utime, _, time_inv, _ = unique_time(timestamps)
uuvw, uvw_idx, uvw_inv, uvw_count = unique_uvw(uvw)

# indeces to recostruct original arrays 
for idx in bl_inv:
    print(ubl[idx])
for idx in time_inv:
    print(utime[idx])

nbl = ubl.shape[0]
ntime = utime.shape[0]

sentinel = np.finfo(timestamps.dtype).max
out_rows = numba.uint32(0)


print("ubl \n", ubl, ubl.shape)
print("bl_idx \n", bl_idx, bl_idx.shape)
print("bl_inv \n", bl_inv, bl_inv.shape)
print("bl_count \n", bl_count, bl_count.shape, bl_count[0])
print("uuvw \n", uuvw, uuvw.shape)
print("uvw_idx \n", uvw_idx, uvw_idx.shape)
print("uvw_inv \n", uvw_inv, uvw_inv.shape)
print("uvw_count \n", uvw_count, uvw_count.shape, uvw_count[0])
# print("utime \n", utime, utime.shape[0])
# print("time_inv \n", time_inv)

scratch = np.full(3*nbl*ntime, -1, dtype=np.int32)
print("scratch \n", scratch, scratch.shape)
row_lookup = scratch[:nbl*ntime].reshape(nbl, ntime)
print("scratch[:nbl*ntime] \n", scratch[:nbl*ntime])
print("row_lookup \n", row_lookup)
bin_lookup = scratch[nbl*ntime:2*nbl*ntime].reshape(nbl, ntime)
print("scratch[nbl*ntime:2*nbl*ntime] \n", scratch[nbl*ntime:2*nbl*ntime])
print("bin_lookup \n", bin_lookup)
inv_argsort = scratch[2*nbl*ntime:]
print("inv_argsort \n", inv_argsort)
time_lookup = np.zeros((nbl, ntime), dtype=timestamps.dtype)
print("time_lookup \n", time_lookup)      
interval_lookup = np.zeros((nbl, ntime), dtype=interval.dtype)
print("interval_lookup \n", interval_lookup)
bin_flagged = np.zeros((nbl, ntime), dtype=np.bool_)
print("bin_flagged \n", bin_flagged)

uvw_lookup = np.zeros((nbl, ntime), dtype=uvw.dtype)
print("uvw_lookup \n", uvw_lookup, uvw_lookup.shape)


print("bl_inv \n", bl_inv)
print("time_inv \n", time_inv)
for r in range(uvw.shape[0]):
    bl = bl_inv[r]
    t = time_inv[r]
    row_lookup[bl, t] = r
    #print("row_lookup \n", row_lookup)
print("row_lookup \n", row_lookup)

# How Do I use row_lookup for find original values ?
print("lets see")
for idx in range(row_lookup.size):
    print(timestamps[idx], interval[idx])
    print(ant1[idx], ant2[idx])

print("verify")
for idx in row_lookup:
    print(timestamps[idx], interval[idx])
    print(ant1[idx], ant2[idx])
print(ant1, ant2)
    
print("\n constructing bin_lookup and time_lookup \n")
# Average times over each baseline and construct the
# bin_lookup and time_lookup arrays
for bl in range(ubl.shape[0]):
    tbin = numba.int32(0)
    bin_count = numba.int32(0)
    bin_flag_count = numba.int32(0)
    bin_low = timestamps.dtype.type(0)

    print('tbin', tbin)
    print('bin_count', bin_count)
    print('bin_flag_count', bin_flag_count)
    print('bin_low', bin_low)
    
    # range of baseline element count
    for t in range(bl_count[0]):
        # Lookup input row
        r = row_lookup[bl, t]
        print("bl : ", bl, "t : ", t, "r = ", r)
        # Ignore if not present
        if r == -1:
            continue

            
#         half_int = interval[r] * 0.5
#         print("half_int",half_int)
#         if bin_count == 0:
#             bin_low = timestamps[r] - half_int
#             print("bin_count ", bin_count)
#             print("bin_low", bin_low)
#         elif timestamps[r] + half_int - bin_low > bl_dist_time_bins[bl]:
#             print("bl_dist_time_bins is less", bl_dist_time_bins[bl], "then ", (timestamps[r] + half_int - bin_low))
#             print("timestapms[r]", timestamps[r], "half_int", half_int,"bin_low", bin_low, "bl_dist_time_bins",bl_dist_time_bins[bl])
            
#             if bin_count > 0:
#                 time_lookup[bl, tbin] /= bin_count
#                 bin_flagged[bl, tbin] = bin_count == bin_flag_count
#             # There was nothing in the bin
#             else:
#                 time_lookup[bl, tbin] = sentinel
#                 bin_flagged[bl, tbin] = False

#             tbin += 1
#             bin_count = 0
#             bin_flag_count = 0

#          I need to code something that replaces the above    
        if bin_count >= bl_dist_time_bins[bl]:
            # Reset the bin_count
            # increase tbin counts
            bin_count = 0
            tbin += 1
            
        bin_lookup[bl, t] = tbin
        
        time_lookup[bl, tbin] += timestamps[r]
        interval_lookup[bl, tbin] += interval[r]
        bin_count += 1
        
        print('tbin', tbin)
        print('bin_count', bin_count)
    
    if bin_count > 0:
        print("******** bin_count > 0 ************")
        time_lookup[bl, tbin] /= bin_count
        bin_flagged[bl, tbin] = bin_count == bin_flag_count
        tbin += 1
    
    out_rows += tbin
    print("out_rows ", out_rows)
    
    for b in range(tbin, ntime):
        time_lookup[bl, b] = sentinel
        bin_flagged[bl, b] = False
    
    print("bin_lookup \n", bin_lookup)
    print("time_lookup \n", time_lookup)
    
print("At The End")
print("row_lookup \n", row_lookup)
print("time_lookup \n", time_lookup)
for idx in row_lookup:
    print(timestamps[idx])
print("bin_lookup \n", bin_lookup)

print("Trying something")
for idx in row_lookup[0]: 
    print(timestamps[idx])
print("Trying something")
for val in time_lookup[0]:
    print(val)
    
flat_time = time_lookup.ravel()
print("flat_time ravel\n", flat_time)
flat_int = interval_lookup.ravel()
print("flat_int ravel\n", flat_int)
argsort = np.argsort(flat_time, kind='mergesort')
print("argsort (mergesort flat_time)\n", argsort)

for i, a in enumerate(argsort):
    inv_argsort[a] = i
print("inv_argsort\n", inv_argsort)

row_map = np.empty((timestamps.shape[0]), dtype=np.uint32)

for in_row in range(timestamps.shape[0]):
    # Lookup baseline and time
    bl = bl_inv[in_row]
    t = time_inv[in_row]

    # lookup time bin and output row
    tbin = bin_lookup[bl, t]
    # lookup output row in inv_argsort
    out_row = inv_argsort[bl*ntime + tbin]

    if out_row >= out_rows:
        raise RowMapperError("out_row >= out_rows")

    # Handle output row flagging
#     set_flag_row(flag_row, in_row,
#                 out_flag_row, out_row,
#                 bin_flagged[bl, tbin])

    row_map[in_row] = out_row

    time_ret = flat_time[argsort[:out_rows]]
    int_ret = flat_int[argsort[:out_rows]]
    
print("Row Map \n", row_map)
print("Time Ret \n", time_ret)
print("Int Ret \n", int_ret)


    




####################################
### Interface with bda averager  ###
####################################

#bda = bd_averager.bda_time(scan, desc, uvw, flag_row, flag, ant1, ant2, interval, exposure, timestamps, time_centroid, vis_data, weights, nbr_bins_av_longest_baseline=1, starttimebin_index=0, nbr_timebins=None, ew=False)

#scan_output, desc_output, flag_row_output, uvw_output, flag_output, ant1_output, ant2_output, interval_output, exposure_output, timestamps_output, time_centroid_output, vis_data_output, weights_output = bda


#################################
### Print Arrays shapes again ###
#################################

# print('------------------\nCompressed Arrays\n------------------')
# print('Desc Output')
# print(desc_output.shape)
# print('UVW Output')
# print(uvw_output.shape)
# print('Flag Output')
# print(flag_output.shape)
# print('Flag Row Output')
# print(flag_row_output.shape)
# print('Antenna 1 Output')
# print(ant1_output.shape)
# print('Antenna 2 Output')
# print(ant2_output.shape)
# print('Interval Output')
# print(interval_output.shape)
# print('Exposure Output')
# print(exposure_output.shape)
# print('Time Output')
# print(timestamps_output.shape)
# print('DATA Output')
# print(vis_data_output.shape)
# print('Time Centroid Output')
# print(time_centroid_output.shape)
# print('Weights Output')
# print(weights_output.shape)
# print('Scan Output')
# print(scan_output.shape)
# print('UVW Things')
# print(uvw_output[:])