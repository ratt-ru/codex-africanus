# -*- coding: utf-8 -*-

from functools import reduce

import numpy as np
from africanus.util.docs import DocstringTemplate
from africanus.util.numba import njit


@njit(nogil=True, cache=True)
def freq_grid_interp(frequency, beam_freq_map):
    # Interpolated grid coordinate
    beam_nud = beam_freq_map.shape[0]
    freq_data = np.empty((frequency.shape[0], 3),
                         dtype=frequency.dtype)

    for f in range(frequency.shape[0]):
        freq = frequency[f]
        lower = 0
        upper = beam_nud - 1

        while lower <= upper:
            mid = lower + (upper - lower) // 2
            beam_freq = beam_freq_map[mid]

            if beam_freq < freq:
                lower = mid + 1
            elif beam_freq > freq:
                upper = mid - 1
            else:
                lower = mid
                break

        # This handles the lower <= upper in the while loop
        lower = min(lower, upper)
        upper = lower + 1

        # Set up scaling, lower weight, lower grid pos
        if lower == -1:
            freq_data[f, 0] = freq / beam_freq_map[0]
            freq_data[f, 1] = 1.0
            freq_data[f, 2] = 0
        elif upper == beam_nud:
            freq_data[f, 0] = freq / beam_freq_map[beam_nud - 1]
            freq_data[f, 1] = 0.0
            freq_data[f, 2] = beam_nud - 2
        else:
            freq_data[f, 0] = 1.0
            freq_low = beam_freq_map[lower]
            freq_high = beam_freq_map[upper]
            freq_diff = freq_high - freq_low
            freq_data[f, 1] = (freq_high - freq) / freq_diff
            freq_data[f, 2] = lower

    return freq_data


@njit(nogil=True, cache=True)
def beam_cube_dde(beam, beam_lm_extents, beam_freq_map,
                  lm, parallactic_angles, point_errors, antenna_scaling,
                  frequency):

    nsrc = lm.shape[0]
    ntime, nants = parallactic_angles.shape
    nchan = frequency.shape[0]
    beam_lw, beam_mh, beam_nud = beam.shape[:3]
    corrs = beam.shape[3:]

    if beam_lw < 2 or beam_mh < 2 or beam_nud < 2:
        raise ValueError("beam_lw, beam_mh and beam_nud must be >= 2")

    # Flatten correlations
    ncorrs = reduce(lambda x, y: x*y, corrs, 1)

    lower_l, upper_l = beam_lm_extents[0]
    lower_m, upper_m = beam_lm_extents[1]

    ex_dtype = beam_lm_extents.dtype

    # Maximum l and m indices in float and int
    lmaxf = ex_dtype.type(beam_lw - 1)
    mmaxf = ex_dtype.type(beam_mh - 1)
    lmaxi = beam_lw - 1
    mmaxi = beam_mh - 1

    lscale = lmaxf / (upper_l - lower_l)
    mscale = mmaxf / (upper_m - lower_m)

    one = ex_dtype.type(1)
    zero = ex_dtype.type(0)

    # Flatten the beam on correlation
    fbeam = beam.reshape((beam_lw, beam_mh, beam_nud, ncorrs))

    # Allocate output array with correlations flattened
    fjones = np.empty((nsrc, ntime, nants, nchan, ncorrs), dtype=beam.dtype)

    # Compute frequency interpolation stuff
    freq_data = freq_grid_interp(frequency, beam_freq_map)

    corr_sum = np.zeros((ncorrs,), dtype=beam.dtype)
    absc_sum = np.zeros((ncorrs,), dtype=beam.real.dtype)
    beam_scratch = np.zeros((ncorrs,), dtype=beam.dtype)

    for t in range(ntime):
        for a in range(nants):
            sin_pa = np.sin(parallactic_angles[t, a])
            cos_pa = np.cos(parallactic_angles[t, a])

            for s in range(nsrc):
                # Extract lm coordinates
                l, m = lm[s]

                for f in range(nchan):
                    # Unpack frequency data
                    freq_scale = freq_data[f, 0]
                    # lower and upper frequency weights
                    nud = freq_data[f, 1]
                    inv_nud = 1.0 - nud
                    # lower and upper frequency grid position
                    gc0 = np.int32(freq_data[f, 2])
                    gc1 = gc0 + 1

                    # Apply any frequency scaling
                    sl = l * freq_scale
                    sm = m * freq_scale

                    # Add pointing errors
                    tl = sl + point_errors[t, a, f, 0]
                    tm = sm + point_errors[t, a, f, 1]

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

                    # Snap to the upper grid coordinates
                    gl1 = min(gl0 + 1, lmaxi)
                    gm1 = min(gm0 + 1, mmaxi)

                    # Difference between grid and offset coordinates
                    ld = vl - gl0
                    md = vm - gm0

                    # Zero accumulation arrays
                    corr_sum[:] = 0
                    absc_sum[:] = 0

                    # Accumulate lower cube correlations
                    beam_scratch[:] = fbeam[gl0, gm0, gc0, :]
                    weight = (one - ld)*(one - md)*nud

                    for c in range(ncorrs):
                        absc_sum[c] += weight * np.abs(beam_scratch[c])
                        corr_sum[c] += weight * beam_scratch[c]

                    beam_scratch[:] = fbeam[gl1, gm0, gc0, :]
                    weight = ld*(one - md)*nud

                    for c in range(ncorrs):
                        absc_sum[c] += weight * np.abs(beam_scratch[c])
                        corr_sum[c] += weight * beam_scratch[c]

                    beam_scratch[:] = fbeam[gl0, gm1, gc0, :]
                    weight = (one - ld)*md*nud

                    for c in range(ncorrs):
                        absc_sum[c] += weight * np.abs(beam_scratch[c])
                        corr_sum[c] += weight * beam_scratch[c]

                    beam_scratch[:] = fbeam[gl1, gm1, gc0, :]
                    weight = ld*md*nud

                    for c in range(ncorrs):
                        absc_sum[c] += weight * np.abs(beam_scratch[c])
                        corr_sum[c] += weight * beam_scratch[c]

                    # Accumulate upper cube correlations
                    beam_scratch[:] = fbeam[gl0, gm0, gc1, :]
                    weight = (one - ld)*(one - md)*inv_nud

                    for c in range(ncorrs):
                        absc_sum[c] += weight * np.abs(beam_scratch[c])
                        corr_sum[c] += weight * beam_scratch[c]

                    beam_scratch[:] = fbeam[gl1, gm0, gc1, :]
                    weight = ld*(one - md)*inv_nud

                    for c in range(ncorrs):
                        absc_sum[c] += weight * np.abs(beam_scratch[c])
                        corr_sum[c] += weight * beam_scratch[c]

                    beam_scratch[:] = fbeam[gl0, gm1, gc1, :]
                    weight = (one - ld)*md*inv_nud

                    for c in range(ncorrs):
                        absc_sum[c] += weight * np.abs(beam_scratch[c])
                        corr_sum[c] += weight * beam_scratch[c]

                    beam_scratch[:] = fbeam[gl1, gm1, gc1, :]
                    weight = ld*md*inv_nud

                    for c in range(ncorrs):
                        absc_sum[c] += weight * np.abs(beam_scratch[c])
                        corr_sum[c] += weight * beam_scratch[c]

                    for c in range(ncorrs):
                        # Added all correlations, normalise
                        div = np.abs(corr_sum[c])

                        if div == 0.0:
                            # This case probably works out to a zero assign
                            corr_sum[c] *= absc_sum[c]
                        else:
                            corr_sum[c] *= absc_sum[c] / div

                    # Assign normalised values
                    fjones[s, t, a, f, :] = corr_sum

    return fjones.reshape((nsrc, ntime, nants, nchan) + corrs)


BEAM_CUBE_DOCS = DocstringTemplate(
    r"""
    Evaluates Direction Dependent Effects along a source's path
    by interpolating the values  of a complex beam cube
    at the source location.

    Notes
    -----
    1. Sources are clamped to the provided `beam_lm_extents`.
    2. Frequencies outside the cube (i.e. outside beam_freq_map)
       introduce linear scaling to the lm coordinates of a source.

    Parameters
    ----------
    beam : $(array_type)
        Complex beam cube of
        shape :code:`(beam_lw, beam_mh, beam_nud, corr, corr)`.
        `beam_lw`, `beam_mh` and `beam_nud` define the size
        of the cube in the l, m and frequency dimensions, respectively.
    beam_lm_extents : $(array_type)
        lm extents of the beam cube of shape :code:`(2, 2)`.
        ``[[lower_l, upper_l], [lower_m, upper_m]]``.
    beam_freq_map : $(array_type)
        Beam frequency map of shape :code:`(beam_nud,)`.
        This array is used to define interpolation along
        the :code:`(chan,)` dimension.
    lm : $(array_type)
        Source lm coordinates of shape :code:`(source, 2)`.
        These coordinates are:

            1. Scaled if the associated frequency lies outside the beam cube.
            2. Offset by pointing errors: ``point_errors``
            3. Rotated by parallactic angles: ``parallactic_angles``.
            4. Scaled by antenna scaling factors: ``antenna_scaling``.

    parallactic_angles : $(array_type)
        Parallactic angles of shape :code:`(time, ant)`.
    point_errors : $(array_type)
        Pointing errors of shape :code:`(time, ant, chan, 2)`.
    antenna_scaling : $(array_type)
        Antenna scaling factors of shape :code:`(ant, chan, 2)`
    frequency : $(array_type)
        Frequencies of shape :code:`(chan,)`.

    Returns
    -------
    ddes : $(array_type)
        Direction Dependent Effects of shape
        :code:`(source, time, ant, chan, corr, corr)`
    """)


try:
    beam_cube_dde.__doc__ = BEAM_CUBE_DOCS.substitute(
                                array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass
