# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numba

from africanus.averaging.support import unique_time, unique_baselines
from africanus.util.numba import is_numba_type_none


def _is_flagged_factory(have_flag_row):
    if have_flag_row:
        def impl(flag_row, r):
            return flag_row[r] != 0
    else:
        def impl(flag_row, r):
            return False

    return numba.njit(nogil=True, cache=True)(impl)


@numba.generated_jit(nopython=True, nogil=True, cache=True)
def row_mapper(time_centroid, exposure, antenna1, antenna2,
               flag_row=None, time_bin_secs=1):
    """
    Generates a mapping from a high resolution row index to
    a low resolution row index in support of time and channel
    averaging code. The `time_centroid` and `exposure` columns
    are also respectively averaged and summed
    in the process of creating the mapping.

    In order to average a chunk of row data, it is necessary to
    group each row (or sample) by baseline and then average
    the time centroid samples present in each baseline in bins of
    `time_bin_secs`. The algorithm is robust in the presence
    of missing time and baseline data.

    The algorithm works as follows:

    1. `time_centroid`, `exposure`, `antenna1` and `antenna2`
    are used to construct a `row_lookup` array of shape `(ubl, utime)`
    mapping a baseline and time_centroid to a row of input data.

    2. For each baseline, `time_bin_secs` times are averaged together
    into a `time_lookup` array of shape `(ubl, utime)`.
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

        row_map, tc_avg, exp_sum = row_mapper(time_centroid,
                                              exposure,
                                              ant1, ant2,
                                              time_bin_secs=3)

        # Recompute time average using row map
        new_time_avg = np.zeros_like(time_avg)
        ant1_avg = np.empty(time_avg.shape, ant1.dtype)
        ant2_avg = np.empty(time_avg.shape, ant2.dtype)
        counts = np.empty(time_avg.shape, np.uint32)

        # Add time and 1 at row_map indices to new_time_avg and counts
        np.add.at(new_time_avg, row_map, time)
        np.add.at(counts, row_map, 1)
        # Normalise
        new_time_avg /= count

        np.testing.assert_array_equal(time_avg, new_time_avg)

        # We assign baselines because each input baseline
        # is mapped to the same output baseline
        ant1_avg[row_map] = ant1
        ant2_avg[row_map] = ant2

    Parameters
    ----------
    time_centroid : :class:`numpy.ndarray`
        Time Centroid values of shape :code:`(row,)`.
    exposure : :class:`numpy.ndarray`
        Exposure times of shape :code:`(row,)`.
    antenna1 : :class:`numpy.ndarray`
        Antenna 1 values of shape :code:`(row,)`.
    antenna2 : :class:`numpy.ndarray`
        Antenna 2 values of shape :code:`(row,)`.
    flag_row : :class:`numpy.ndarray`, optional
        Positive values indicate that a row is flagged, while
        zero implies unflagged. Has shape :code:`(row,)`.
    time_bin_secs : int, optional
        Number of timesteps to average into each bin

    Returns
    -------
    row_lookup : :class:`numpy.ndarray`
        Mapping from `np.arange(row)` to output row indices
    time_centroid_avg : :class:`numpy.ndarray`
        Averaged time values of shape :code:`(out_row,)`
    exposure_sum : :class:`numpy.ndarray`
        Summed exposure values of shape :code:`(out_row,)`

    """
    have_flag_row = not is_numba_type_none(flag_row)
    is_flagged_fn = _is_flagged_factory(have_flag_row)

    def impl(time_centroid, exposure, antenna1, antenna2,
             flag_row=None, time_bin_secs=1):
        ubl, _, bl_inv, _ = unique_baselines(antenna1, antenna2)
        utime, _, time_inv, _ = unique_time(time_centroid)

        nbl = ubl.shape[0]
        ntime = utime.shape[0]

        sentinel = np.finfo(time_centroid.dtype).max
        out_rows = numba.uint32(0)

        scratch = np.full(3*nbl*ntime, -1, dtype=np.int32)
        row_lookup = scratch[:nbl*ntime].reshape(nbl, ntime)
        bin_lookup = scratch[nbl*ntime:2*nbl*ntime].reshape(nbl, ntime)
        inv_argsort = scratch[2*nbl*ntime:]
        time_lookup = np.zeros((nbl, ntime), dtype=time_centroid.dtype)
        exposure_lookup = np.zeros((nbl, ntime), dtype=exposure.dtype)

        for r in range(time_centroid.shape[0]):
            bl = bl_inv[r]
            t = time_inv[r]
            row_lookup[bl, t] = r

        # Average times over each baseline and construct the
        # bin_lookup and time_lookup arrays
        for bl in range(ubl.shape[0]):
            tbin = numba.int32(0)
            bin_count = numba.int32(0)
            bin_low = time_centroid.dtype.type(0)

            for t in range(utime.shape[0]):
                # Lookup input row
                r = row_lookup[bl, t]

                # Ignore if not present
                if r == -1:
                    continue

                # Indirectly map row to the current bin
                bin_lookup[bl, t] = tbin

                # If the row is flagged, ignore any contributions
                # by time_centroid and exposure
                if is_flagged_fn(flag_row, r):
                    continue

                half_exp = exposure[r] * 0.5

                # At this point, we decide whether to contribute to
                # the current bin, or create a new one. We don't add
                # the current sample to the current bin if
                # high - low >= time_bin_secs

                # We're starting a new bin anyway,
                # just set the lower bin value
                # and ignore normalisation
                if bin_count == 0:
                    bin_low = time_centroid[r] - half_exp
                # If we exceed the seconds in the bin,
                # normalise the centroid and start a new bin
                elif time_centroid[r] + half_exp - bin_low > time_bin_secs:
                    time_lookup[bl, tbin] /= bin_count
                    tbin += 1
                    bin_count = 0
                    # The current sample now contributes to the next bin
                    bin_lookup[bl, t] = tbin

                # Add sample to the bin and increment the count
                time_lookup[bl, tbin] += time_centroid[r]
                exposure_lookup[bl, tbin] += exposure[r]
                bin_count += 1

            # Normalise centroid in the last bin if necessary
            if bin_count > 0:
                time_lookup[bl, tbin] /= bin_count
                tbin += 1

            # Add this baseline's number of bins to the output rows
            out_rows += tbin

            # Set any remaining bins to sentinel value
            for b in range(tbin, ntime):
                time_lookup[bl, b] = sentinel

        # Flatten the time lookup and argsort it
        flat_time = time_lookup.ravel()
        argsort = np.argsort(flat_time, kind='mergesort')

        # Generate lookup from flattened (bl, time) to output row
        for i, a in enumerate(argsort):
            inv_argsort[a] = i

        # Number of flagged rows
        total_input_rows = time_centroid.shape[0]
        flagged_rows = 0

        for r in range(total_input_rows):
            if is_flagged_fn(flag_row, r):
                flagged_rows += 1

        map_rows = total_input_rows - flagged_rows

        # Construct the final row map
        row_map = np.empty((2, map_rows), dtype=np.uint32)

        # Running count of encountered flags
        flag_count = numba.int32(0)

        # foreach input row
        for in_row in range(time_centroid.shape[0]):
            # Lookup baseline and time
            bl = bl_inv[in_row]
            t = time_inv[in_row]

            # Ignore if flagged and increase flag_count
            if is_flagged_fn(flag_row, in_row):
                flag_count += 1
                continue

            # lookup time bin and output row
            tbin = bin_lookup[bl, t]
            out_row = inv_argsort[bl*ntime + tbin]

            # lookup output row in inv_argsort
            row_map[0, in_row - flag_count] = in_row
            row_map[1, in_row - flag_count] = out_row

        return row_map, flat_time[argsort[:out_rows]]

    return impl
