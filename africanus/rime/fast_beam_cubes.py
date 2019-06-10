#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from functools import reduce
except ImportError:
    pass

import numpy as np
from africanus.util.numba import njit


@njit(nogil=True, cache=True)
def freq_grid_interp(frequencies, beam_freq_map):
    # Interpolated grid coordinate
    grid_pos_f = np.interp(frequencies, beam_freq_map,
                           np.arange(beam_freq_map.size))
    # Floor grid position
    grid_pos = np.empty(grid_pos_f.shape, dtype=np.int32)
    # Frequency scaling for cases when we're below or above the beam
    freq_scale = np.empty_like(frequencies)
    # Frequency difference between the channel frequency and the grid frequency
    freq_grid_diff = np.empty(frequencies.shape, dtype=frequencies.dtype)

    beam_nud = beam_freq_map.shape[0]
    beam_nud_f_minus_one = grid_pos_f.dtype.type(beam_freq_map.shape[0] - 1)

    for chan in range(frequencies.shape[0]):
        freq = frequencies[chan]

        # Below the beam_freq_map, we'll clamp frequencies
        # to the lower position and introduce scaling for the lm coordinates
        if freq < beam_freq_map[0]:
            freq_scale[chan] = freq / beam_freq_map[0]
            grid_pos[chan] = 0
            freq = beam_freq_map[0]
        # Above, or exactly on the last value of the beam_freq_map
        # clamp frequencies to the upper position and introduce
        # scaling for lm coordinates
        elif grid_pos_f[chan] == beam_nud_f_minus_one:
            if freq > beam_freq_map[-1]:
                freq_scale[chan] = freq / beam_freq_map[-1]
            else:
                freq_scale[chan] = 1.0

            # Shift one grid point back from the end
            grid_pos[chan] = beam_nud - 2
            freq = beam_freq_map[-1]
        # Standard case
        else:
            grid_pos[chan] = np.floor(grid_pos_f[chan])
            freq_scale[chan] = 1.0

        # Difference between the frequency and the floor grid position freq
        freq_grid_diff[chan] = freq - beam_freq_map[grid_pos[chan]]

    return grid_pos, freq_scale, freq_grid_diff


@njit(nogil=True, cache=True)
def beam_cube_dde(beam, beam_lm_extents, beam_freq_map,
                  lm, parangles, point_errors, antenna_scaling,
                  frequencies):

    nsrc = lm.shape[0]
    ntime, nants = parangles.shape
    nchan = frequencies.shape[0]
    beam_lw, beam_mh, beam_nud = beam.shape[:3]
    corrs = beam.shape[3:]

    if beam_lw < 0 or beam_mh < 0 or beam_nud < 0:
        raise ValueError("beam_lw, beam_mh and beam_nud must be >= 2")

    # Flatten correlations
    ncorrs = reduce(lambda x, y: x*y, corrs, 1)

    lower_l, upper_l = beam_lm_extents[0]
    lower_m, upper_m = beam_lm_extents[1]

    ex_dtype = beam_lm_extents.dtype

    # Maximum l and m indices in float and int
    lmaxf = ex_dtype.type(beam_lw - 1)
    mmaxf = ex_dtype.type(beam_mh - 1)

    lscale = lmaxf / (upper_l - lower_l)
    mscale = mmaxf / (upper_m - lower_m)

    one = ex_dtype.type(1)
    zero = ex_dtype.type(0)

    # Flatten the beam on correlation
    fbeam = beam.reshape((beam_lw, beam_mh, beam_nud, ncorrs))

    # Allocate output array with correlations flattened
    fjones = np.empty((nsrc, ntime, nants, nchan, ncorrs), dtype=beam.dtype)

    # Compute frequency interpolation stuff
    grid_pos, freq_scale, freq_diff = freq_grid_interp(frequencies,
                                                       beam_freq_map)

    corr_sum = np.zeros((ncorrs,), dtype=beam.dtype)
    absc_sum = np.zeros((ncorrs,), dtype=beam.real.dtype)
    beam_scratch = np.zeros((ncorrs,), dtype=beam.dtype)
    weight = np.zeros((8,), dtype=ex_dtype)

    for t in range(ntime):
        for a in range(nants):
            sin_pa = np.sin(parangles[t, a])
            cos_pa = np.cos(parangles[t, a])

            for s in range(nsrc):
                # Extract lm coordinates
                l, m = lm[s]

                for f in range(nchan):
                    # Apply any frequency scaling
                    sl = l * freq_scale[f]
                    sm = m * freq_scale[f]

                    # Add pointing errors
                    tl = sl + point_errors[t, a, 0]
                    tm = sm + point_errors[t, a, 1]

                    # Rotate lm coordinate angle
                    vl = tl*cos_pa - tm*sin_pa
                    vm = tl*sin_pa + tm*cos_pa

                    # Scale by antenna scaling
                    vl *= antenna_scaling[a, f, 0]
                    vm *= antenna_scaling[a, f, 1]

                    # Shift into the cube coordinate system
                    vl = lscale*(vl - lower_l)
                    vm = mscale*(vm - lower_m)

                    # Clamp the coordinates to the edges of the cube
                    vl = max(zero, min(vl, lmaxf))
                    vm = max(zero, min(vm, mmaxf))

                    # Snap to the lower grid coordinates
                    gl0 = np.int32(np.floor(vl))
                    gm0 = np.int32(np.floor(vm))
                    gc0 = grid_pos[f]

                    # Snap to the upper grid coordinates
                    gc1 = gc0 + 1

                    # Difference between grid and offset coordinates
                    ld = vl - gl0
                    md = vm - gm0
                    nud = freq_diff[f]

                    # Zero accumulation arrays
                    corr_sum[:] = 0
                    absc_sum[:] = 0

                    # Precompute lower cube weights
                    weight[0] = (one - ld)*(one - md)*nud
                    weight[1] = ld*(one - md)*nud
                    weight[2] = (one - ld)*md*nud
                    weight[3] = ld*md*nud

                    # Precompute upper cube weights
                    weight[4] = (one - ld)*(one - md)*(one - nud)
                    weight[5] = ld*(one - md)*(one - nud)
                    weight[6] = (one - ld)*md*(one - nud)
                    weight[7] = ld*md*(one - nud)

                    # Accumulate lower cube correlations
                    beam_scratch[:] = fbeam[gl0, gm0, gc0]

                    for c in range(ncorrs):
                        absc_sum[c] += weight[0] * np.abs(beam_scratch[c])
                        corr_sum[c] += weight[0] * beam_scratch[c]

                        absc_sum[c] += weight[1] * np.abs(beam_scratch[c])
                        corr_sum[c] += weight[1] * beam_scratch[c]

                        absc_sum[c] += weight[2] * np.abs(beam_scratch[c])
                        corr_sum[c] += weight[2] * beam_scratch[c]

                        absc_sum[c] += weight[3] * np.abs(beam_scratch[c])
                        corr_sum[c] += weight[3] * beam_scratch[c]

                    # Accumulate upper cube correlations
                    beam_scratch[:] = fbeam[gl0, gm0, gc1]

                    for c in range(ncorrs):
                        absc_sum[c] += weight[4] * np.abs(beam_scratch[c])
                        corr_sum[c] += weight[4] * beam_scratch[c]

                        absc_sum[c] += weight[5] * np.abs(beam_scratch[c])
                        corr_sum[c] += weight[5] * beam_scratch[c]

                        absc_sum[c] += weight[6] * np.abs(beam_scratch[c])
                        corr_sum[c] += weight[6] * beam_scratch[c]

                        absc_sum[c] += weight[7] * np.abs(beam_scratch[c])
                        corr_sum[c] += weight[7] * beam_scratch[c]

                        # Added all correlations, normalise
                        norm = absc_sum[c] / np.abs(corr_sum[c])
                        corr_sum[c] *= absc_sum[c] if np.isinf(norm) else norm

                    # Assign normalised values
                    fjones[s, t, a, f, :] = corr_sum

    return fjones.reshape((nsrc, ntime, nants, nchan) + corrs)
