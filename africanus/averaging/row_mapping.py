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


def _row_or_minus_one_factory(have_flag_row):
    if have_flag_row:
        def impl(flag_row, r):
            return r if flag_row[r] == 0 else -1
    else:
        def impl(flag_row, r):
            return r

    return numba.njit(nogil=True, cache=True)(impl)


@numba.generated_jit(nopython=True, nogil=True, cache=True)
def row_mapper(time, antenna1, antenna2, flag_row=None, time_bin_size=1):
    """
    Generates a mapping from a high resolution row index to
    a low resolution row index in support of time and channel
    averaging code. The `time` column is also averaged during this
    process in order to produce the mapping.

    In order to average a chunk of row data, it is necessary to
    group each row (or sample) by baseline and then average
    the time samples present in each baseline in bins of
    `time_bin_size`. The algorithm is robust in the presence
    of missing time and baseline data.

    The algorithm works as follows:

    1. `antenna1`, `antenna2` and `time`are used to construct a
    `row_lookup` array of shape `(ubl, utime)` mapping a baseline and time
    to a row of input data.

    2. For each baseline, `time_bin_size` times are averaged together
    into a `time_lookup` array of shape `(ubl, tbins)`.
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

        row_map, time_avg = row_mapper(time, ant1, ant2, time_bin_size=3)

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
    time : :class:`numpy.ndarray`
        Time values of shape :code:`(row,)`
    antenna1 : :class:`numpy.ndarray`
        Antenna 1 values of shape :code:`(row,)`
    antenna2 : :class:`numpy.ndarray`
        Antenna 2 values of shape :code:`(row,)`
    flag_row : :class:`numpy.ndarray`, optional
        Positive values indicate that a row is flagged, while
        zero implies unflagged. Has shape :code:`(row,)`.
    time_bin_size : int, optional
        Number of timesteps to average into each bin

    Returns
    -------
    row_lookup : :class:`numpy.ndarray`
        Mapping from `np.arange(row)` to output row indices
    time_avg : :class:`numpy.ndarray`
        Averaged time values of shape :code:`(out_row,)`
    """
    have_flag_row = not is_numba_type_none(flag_row)
    row_or_minus_one_fn = _row_or_minus_one_factory(have_flag_row)
    is_flagged_fn = _is_flagged_factory(have_flag_row)

    def impl(time, antenna1, antenna2, flag_row=None, time_bin_size=1):
        ubl, bl_inv, bl_counts = unique_baselines(antenna1, antenna2)
        utime, time_inv, time_counts = unique_time(time)

        nbl = ubl.shape[0]
        ntime = utime.shape[0]
        tbins = (ntime + time_bin_size - 1) // time_bin_size

        sentinel = np.finfo(time.dtype).max
        out_rows = 0

        scratch = np.empty(2*nbl*ntime + nbl*tbins, dtype=np.intp)
        row_lookup = scratch[:nbl*ntime].reshape(nbl, ntime)
        bin_lookup = scratch[nbl*ntime:2*nbl*ntime].reshape(nbl, ntime)
        inv_argsort = scratch[2*nbl*ntime:]
        time_lookup = np.zeros((nbl, tbins), dtype=time.dtype)

        # Construct the row_lookup matrix
        row_lookup[:, :] = -1
        bin_lookup[:, :] = -1

        flagged_rows = 0

        for r in range(time.shape[0]):
            bl = bl_inv[r]
            t = time_inv[r]

            if is_flagged_fn(flag_row, r):
                r = -1
                flagged_rows += 1

            row_lookup[bl, t] = r

        # Average times over each baseline and construct the
        # bin_lookup and time_lookup arrays
        for bl in range(ubl.shape[0]):
            tbin = 0
            bin_count = 0

            for t in range(utime.shape[0]):
                # Lookup input row and ignore if it's not present
                r = row_lookup[bl, t]

                if r == -1:
                    continue

                # Map to the relevant bin
                bin_lookup[bl, t] = tbin

                # Add sample to the bin and increment the count
                time_lookup[bl, tbin] += time[r]
                bin_count += 1

                # Normalise if we've filled a bin
                if bin_count == time_bin_size:
                    time_lookup[bl, tbin] /= bin_count
                    bin_count = 0
                    tbin += 1

            # Normalise the last bin if necessary
            if bin_count > 0:
                time_lookup[bl, tbin] /= bin_count
                tbin += 1

            # Add this baselines number of bins to the output rows
            out_rows += tbin

            # Set any remaining bins to sentinel value
            for b in range(tbin, tbins):
                time_lookup[bl, b] = sentinel

        # Flatten the time lookup and argsort it
        flat_time = time_lookup.ravel()
        argsort = np.argsort(flat_time, kind='mergesort')

        # Generate lookup from flattened (bl, tbin) to output row
        for i, a in enumerate(argsort):
            inv_argsort[a] = i

        # Construct the final row map
        row_map = np.empty((time.shape[0] - flagged_rows, 2), dtype=np.uint32)

        flags_found = 0

        # Foreach input row
        for in_row in range(time.shape[0]):
            # Lookup baseline and time
            bl = bl_inv[in_row]
            t = time_inv[in_row]

            # flagged, ignore input row
            if is_flagged_fn(flag_row, in_row):
                flags_found += 1
                continue

            # Adjust map row for flagged rows
            map_row = in_row - flags_found

            # lookup time bin and output row
            tbin = bin_lookup[bl, t]
            out_row = inv_argsort[bl*tbins + tbin]

            # lookup output row in inv_argsort
            row_map[map_row, 0] = in_row
            row_map[map_row, 1] = out_row

        return row_map, flat_time[argsort[:out_rows]]

    return impl
