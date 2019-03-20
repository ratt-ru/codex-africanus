# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import numpy as np
import numba

from africanus.averaging.support import unique_time, unique_baselines
from africanus.util.numba import is_numba_type_none, generated_jit, njit


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

            out_flag_row[out_row] = flagged
    else:
        def impl(flag_row, in_row, out_flag_row, out_row, flagged):
            pass

    return njit(nogil=True, cache=True)(impl)


RowMapOutput = namedtuple("RowMapOutput",
                          ["map", "time_centroid", "exposure", "flag_row"])


@generated_jit(nopython=True, nogil=True, cache=True)
def row_mapper(time_centroid, exposure, antenna1, antenna2,
               flag_row=None, time_bin_secs=1):
    """
    Generates a mapping from a high resolution row index to
    a low resolution row index in support of time and channel
    averaging code. The `time_centroid` and `exposure` columns
    are also respectively averaged and summed
    in the process of creating the mapping and a
    `flag_row` column is also returned if one is provided.

    In order to average a chunk of row data, it is necessary to
    group each row (or sample) by baseline and then average
    the time centroid samples present in each baseline in bins of
    `time_bin_secs`.

    Flagged data is handled as follows:

    1. It does not contribute to a bin at all if
       there are other unflagged samples in the bin.
    2. It is the only contribution to a bin if
       all samples in the bin are flagged.

    The algorithm is robust in the presence of missing time and baseline data.

    The algorithm works as follows:

    1. `time_centroid`, `exposure`, `antenna1` and `antenna2`
    are used to construct a `row_lookup` array of shape `(ubl, utime)`
    mapping a baseline and time_centroid to a row of input data.

    2. For each baseline, `time_bin_secs` times are averaged together
    into two separate `centroid_lookup` arrays of shape `(ubl, utime)`.
    The first contains the average of unflagged samples, while the
    second contains the average of flagged samples.

    If the bin contains some unflagged samples, the unflagged average
    is used as the bin average, whereas if all samples are flagged
    the flagged average is used.

    Not all bins may be filled for a baseline if data is missing --
    these bins are assigned a sentinel value set to the
    maximum floating point value.

    A secondary `bin_lookup` array of shape `(ubl, utime)` is constructed
    mapping a time in the `row_lookup` array to a
    time bin in `centroid_lookup`.

    3. The `centroid_lookup` array is flattened and argsorted with a stable
    merge sort. As missing values are set to the maximum floating point
    value, this moves valid data to the front and missing data to the back.
    This has the effect of lexicographically sorts the data
    in an ascending `(time, bl)` order

    4. Input rows are then mapped via the `row_lookup`, `bin_lookup`
    and argsorted `centroid_lookup` arrays to an output row.

    .. code-block:: python

        ret = row_mapper(time_centroid, exposure,
                         ant1, ant2, flag_row,
                         time_bin_secs=3)

        # Only add a bin's contribution if both input and output
        # are (a) flagged or (b) unflagged
        sel = flag_row == ret.flag_row[ret.map]
        sel_map = ret.map[sel]

        # Recompute time average using row map
        time_centroid = np.zeros_like(ret.time_centroid)
        ant1_avg = np.empty(time_centroid.shape, ant1.dtype)
        ant2_avg = np.empty(time_centroid.shape, ant2.dtype)
        counts = np.empty(time_centroid.shape, np.uint32)

        # Add time and 1 at map indices to time_centroid and counts
        np.add.at(time_centroid, sel_map, time_centroid[sel])
        np.add.at(counts, sel_map, 1)
        # Normalise
        time_centroid /= count

        np.testing.assert_array_equal(time_centroid, ret.time_centroid)

        # We assign baselines because each input baseline
        # is mapped to the same output baseline
        ant1_avg[sel_map] = ant1[sel]
        ant2_avg[sel_map] = ant2[sel]

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
    map : :class:`numpy.ndarray`
        Mapping from `np.arange(row)` to output row indices
        of shape :code:`(row,)`
    time_centroid : :class:`numpy.ndarray`
        Averaged time values of shape :code:`(out_row,)`
    exposure_sum : :class:`numpy.ndarray`
        Summed exposure values of shape :code:`(out_row,)`
    flag_row : :class:`numpy.ndarray` or None
        Output flag rows of shape :code:`(out_row,)`.
        None if no input flag_row was supplied.

    Raises
    ------
    RowMapperError
        Raised if an illegal condition occurs

    """
    have_flag_row = not is_numba_type_none(flag_row)
    is_flagged_fn = is_flagged_factory(have_flag_row)

    output_flag_row = output_factory(have_flag_row)
    set_flag_row = set_flag_row_factory(have_flag_row)

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
        centroid_lookup = np.zeros((2, nbl, ntime), dtype=time_centroid.dtype)
        exposure_lookup = np.zeros((2, nbl, ntime), dtype=exposure.dtype)

        bin_flagged = np.zeros((nbl, ntime), dtype=np.bool_)

        # Create a mapping from the full bl x time resolution back
        # to the original input rows
        for r in range(time_centroid.shape[0]):
            bl = bl_inv[r]
            t = time_inv[r]
            row_lookup[bl, t] = r

        # Average times over each baseline and construct the
        # bin_lookup and centroid_lookup arrays
        for bl in range(ubl.shape[0]):
            tbin = numba.int32(0)
            bin_count = numba.int32(0)
            bin_flag_count = numba.int32(0)
            bin_low = time_centroid.dtype.type(0)

            for t in range(utime.shape[0]):
                # Lookup input row
                r = row_lookup[bl, t]

                # Ignore if not present
                if r == -1:
                    continue

                # At this point, we decide whether to contribute to
                # the current bin, or create a new one. We don't add
                # the current sample to the current bin if
                # high - low >= time_bin_secs
                half_exp = exposure[r] * 0.5

                # We're starting a new bin anyway,
                # just set the lower bin value
                if bin_count == 0:
                    bin_low = time_centroid[r] - half_exp
                # If we exceed the seconds in the bin,
                # normalise the centroid and start a new bin
                elif time_centroid[r] + half_exp - bin_low > time_bin_secs:
                    # Try normalise using unflagged entries first
                    if bin_count > 0:
                        centroid_lookup[0, bl, tbin] /= bin_count
                        bin_flagged[bl, tbin] = False
                    # else normalise completely flagged bin
                    elif bin_flag_count > 0:
                        centroid_lookup[1, bl, tbin] /= bin_flag_count
                        flagged_centroid = centroid_lookup[1, bl, tbin]
                        flagged_exposure = exposure_lookup[1, bl, tbin]
                        centroid_lookup[0, bl, tbin] = flagged_centroid
                        exposure_lookup[0, bl, tbin] = flagged_exposure
                        bin_flagged[bl, tbin] = True
                    # There was nothing in the bin
                    else:
                        centroid_lookup[0, bl, tbin] = sentinel
                        bin_flagged[bl, tbin] = False

                    tbin += 1
                    bin_count = 0
                    bin_flag_count = 0

                # Record the bin associated with the row
                bin_lookup[bl, t] = tbin

                # Add sample to the appropriate bin and increment the count
                if not is_flagged_fn(flag_row, r):
                    centroid_lookup[0, bl, tbin] += time_centroid[r]
                    exposure_lookup[0, bl, tbin] += exposure[r]
                    bin_count += 1
                else:
                    centroid_lookup[1, bl, tbin] += time_centroid[r]
                    exposure_lookup[1, bl, tbin] += exposure[r]
                    bin_flag_count += 1

            # Normalise if we have unflagged entries
            if bin_count > 0:
                centroid_lookup[0, bl, tbin] /= bin_count
                bin_flagged[bl, tbin] = False
                tbin += 1
            # Normalise if we're completely flagged
            elif bin_flag_count > 0:
                centroid_lookup[1, bl, tbin] /= bin_flag_count
                flagged_centroid = centroid_lookup[1, bl, tbin]
                flagged_exposure = exposure_lookup[1, bl, tbin]
                centroid_lookup[0, bl, tbin] = flagged_centroid
                exposure_lookup[0, bl, tbin] = flagged_exposure
                bin_flagged[bl, tbin] = True
                tbin += 1

            # Add this baseline's number of bins to the output rows
            out_rows += tbin

            # Set any remaining bins to sentinel value and unflagged
            for b in range(tbin, ntime):
                centroid_lookup[0, bl, b] = sentinel
                bin_flagged[bl, b] = False

        # Flatten the time lookup and argsort it
        flat_time = centroid_lookup[0, :, :].ravel()
        flat_exp = exposure_lookup[0, :, :].ravel()
        argsort = np.argsort(flat_time, kind='mergesort')

        # Generate lookup from flattened (bl, time) to output row
        for i, a in enumerate(argsort):
            inv_argsort[a] = i

        # Construct the final row map
        row_map = np.empty((time_centroid.shape[0]), dtype=np.uint32)

        # Construct output flag row, if necessary
        out_flag_row = output_flag_row(out_rows, flag_row)

        # foreach input row
        for in_row in range(time_centroid.shape[0]):
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
            set_flag_row(flag_row, in_row,
                         out_flag_row, out_row,
                         bin_flagged[bl, tbin])

            row_map[in_row] = out_row

        time_ret = flat_time[argsort[:out_rows]]
        exp_ret = flat_exp[argsort[:out_rows]]

        return RowMapOutput(row_map, time_ret, exp_ret, out_flag_row)

    return impl
