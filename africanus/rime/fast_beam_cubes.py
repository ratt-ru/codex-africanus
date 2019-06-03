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
def tri_interp(pol_sum, abs_sum,
               beam, gl,  gm,  gchan,
               beam_lw, beam_mh, beam_nud,
               corr, weight):

    gl = np.int32(gl)
    gm = np.int32(gm)
    gchan = np.int32(gchan)

    if gl < 0 or gl > beam_lw or gm < 0 or gm > beam_mh:
        return

    data = beam[gl, gm, gchan, corr]
    abs_sum += weight * np.abs(data)
    pol_sum += weight * data

    return abs_sum, pol_sum


@njit(nogil=True, cache=True)
def beam_cube_dde(beam, beam_lm_extents, beam_freq_map,
                  lm, parangles, point_errors, antenna_scaling,
                  frequencies):

    nsrc = lm.shape[0]
    ntime, nants = parangles.shape
    nchan = frequencies.shape[0]
    beam_lw, beam_mh, beam_nud = beam.shape[:3]
    corrs = beam.shape[3:]
    # Flatten correlations
    ncorrs = reduce(lambda x, y: x*y, corrs, 1)

    lower_l, upper_l = beam_lm_extents[0]
    lower_m, upper_m = beam_lm_extents[1]

    ex_dtype = beam_lm_extents.dtype

    lscale = (beam_lw - 1) / (upper_l - lower_l)
    mscale = (beam_mh - 1) / (upper_m - lower_m)

    one = ex_dtype.type(1)
    zero = ex_dtype.type(0)

    lmax = ex_dtype.type(beam_lw - 1)
    mmax = ex_dtype.type(beam_mh - 1)
    fmax = beam_nud - 1

    jones = np.empty((nsrc, ntime, nants, nchan, ncorrs))

    print()
    print(beam_freq_map)
    print(beam_freq_map.shape)
    print(corrs, ncorrs)

    fbeam = beam.reshape((beam_lw, beam_mh, beam_nud, ncorrs))

    freq_grid = np.searchsorted(beam_freq_map, frequencies, side='left')
    freq_scale = np.empty_like(frequencies)
    freq_grid_low = np.empty(frequencies.shape, dtype=np.int32)
    freq_grid_hi = np.empty(frequencies.shape, dtype=np.int32)

    for chan in range(nchan):
        freq = frequencies[chan]

        if freq < freq_grid[0]:
            freq_scale[chan] = freq / freq_grid[0]
            freq_grid_low[chan] = 0
            freq_grid_hi[chan] = 1
        elif freq > freq_grid[-1]:
            freq_scale[chan] = freq / freq_grid[-1]
            freq_grid_low[chan] = nchan - 2
            freq_grid_hi[chan] = nchan - 1
        else:
            freq_scale[chan] = 1.0
            freq_grid_low[chan] = freq_grid[chan]
            freq_grid_high[chan] = freq_grid[chan] + 1

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

                    vl = min(vl, lmax)
                    vl = max(zero, vl)
                    vm = min(vm, mmax)
                    vm = max(zero, vm)

                    # Find the snapped grid coordinates
                    gl0 = np.floor(vl)
                    gm0 = np.floor(vm)

                    gl1 = min(gl0 + one, lmax)
                    gm1 = min(gm0 + one, mmax)

                    # Difference between grid and offset coordinates
                    ld = vl - gl0
                    md = vm - gm0

                    for c in range(ncorrs):
                        pol_sum = beam.dtype.type(0)
                        abs_sum = beam.real.dtype.type(0)

                        weight = (one - ld)*(one - md)

                        pol_sum, abs_sum = tri_interp(pol_sum, abs_sum, fbeam,
                                                      gl0, gm0, 0,
                                                      beam_lw, beam_mh,
                                                      beam_nud, c, weight)

                        pol_sum, abs_sum = tri_interp(pol_sum, abs_sum, fbeam,
                                                      gl0, gm0, 0,
                                                      beam_lw, beam_mh,
                                                      beam_nud, c, weight)

                        pol_sum, abs_sum = tri_interp(pol_sum, abs_sum, fbeam,
                                                      gl0, gm0, 0,
                                                      beam_lw, beam_mh,
                                                      beam_nud, c, weight)

                        pol_sum, abs_sum = tri_interp(pol_sum, abs_sum, fbeam,
                                                      gl0, gm0, 0,
                                                      beam_lw, beam_mh,
                                                      beam_nud, c, weight)

                        norm = abs_sum / np.abs(pol_sum)

                        if np.isinf(norm):
                            norm = abs_sum

                        jones.real[s, t, a, f, c] = pol_sum.real * norm
                        jones.imag[s, t, a, f, c] = pol_sum.imag * norm

    return jones
