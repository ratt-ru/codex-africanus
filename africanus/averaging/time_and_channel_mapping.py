# -*- coding: utf-8 -*-


from collections import namedtuple
from numbers import Number

import numpy as np
import numba
from numba.experimental import jitclass

from africanus.averaging.support import unique_time, unique_baselines
from africanus.util.numba import is_numba_type_none, generated_jit, njit


class RowMapperError(Exception):
    pass


def _numba_type(obj):
    if isinstance(obj, np.ndarray):
        return numba.typeof(obj.dtype).dtype
    elif isinstance(obj, numba.types.npytypes.Array):
        return obj.dtype
    elif isinstance(obj, (np.dtype, numba.types.Type)):
        return numba.typeof(obj).dtype
    elif isinstance(obj, Number):
        return numba.typeof(obj)
    else:
        raise TypeError(f"Unhandled type {type(obj)}")


def binner_factory(time, interval, antenna1, antenna2,
                   flag_row, time_bin_secs):
    if flag_row is None:
        have_flag_row = False
    else:
        have_flag_row = not is_numba_type_none(flag_row)

    class Binner:
        def __init__(self, time, interval, antenna1, antenna2,
                     flag_row, time_bin_secs):
            ubl, _, bl_inv, _ = unique_baselines(antenna1, antenna2)
            utime, _, time_inv, _ = unique_time(time)

            ntime = utime.shape[0]
            nbl = ubl.shape[0]
            self.bl_inv = bl_inv
            self.time_inv = time_inv
            self.out_rows = 0
            row_lookup = np.full((nbl, ntime), -1, dtype=np.intp)

            # Create a mapping from the full bl x time resolution back
            # to the original input rows
            for r, (t, bl) in enumerate(zip(time_inv, bl_inv)):
                if row_lookup[bl, t] == -1:
                    row_lookup[bl, t] = r
                else:
                    raise ValueError("Duplicate (TIME, ANTENNA1, ANTENNA2) "
                                     "combinations were discovered in the "
                                     "input data. This is usually caused by "
                                     "not partitioning your data sufficiently "
                                     "by indexing columns, DATA_DESC_ID "
                                     "and SCAN_NUMBER in particular.")

            sentinel = np.finfo(time.dtype).max

            self.row_lookup = row_lookup
            self.time_bin_secs = time_bin_secs
            self.time_lookup = np.full((nbl, ntime), sentinel, time.dtype)
            self.interval_lookup = np.zeros((nbl, ntime), interval.dtype)
            self.bin_flagged = np.full((nbl, ntime), False)
            self.bin_lookup = np.full((nbl, ntime), -1)

            self.time = time
            self.interval = interval

            if have_flag_row:
                self.flag_row = flag_row

        def start_baseline(self):
            self.tbin = 0
            self.rs = 0
            self.re = 0
            self.bin_count = 0
            self.bin_flag_count = 0

        def finalise_baseline(self):
            self.out_rows += self.tbin

        @property
        def bin_empty(self):
            return self.bin_count == 0

        def start_bin(self, row, flag_row):
            self.rs = row
            self.re = row
            self.bin_count = 1
            self.bin_flag_count = int(have_flag_row and flag_row[row] != 0)

        def add_row(self, row, time, interval, flag_row):
            rs = self.rs

            if self.re == row:
                raise ValueError("start_bin should be called "
                                 "to start a bin before add_row "
                                 "is called.")

            dt = ((time[row] + 0.5*interval[row]) -
                  (time[rs] - 0.5*interval[rs]))

            if dt > self.time_bin_secs:
                return False
            self.re = row
            self.bin_count += 1
            flagged = have_flag_row and flag_row[row] != 0
            self.bin_flag_count += int(flagged)

            return True

        def finalise_bin(self, bl, next_row, time, interval):
            rs = self.rs
            re = self.re
            tbin = self.tbin

            # No interpolation required
            if rs == re:
                bin_time = time[rs]
                bin_interval = interval[rs]
            else:
                # Interpolate between bin start and end times.

                # 1. We use the previous bin and the first row of
                #    the next bin to establish this where possible
                #    as these points determine the full bin extent.
                # 2. Otherwise we must use the first and last row
                #    in the bin for the first and last bin
                #    respectively. This is not as accurate as (1),
                #    but is best effort in the case of missing edge data
                if tbin > 0:
                    # Use the bin prior to this one to establish
                    # the start time of the bin
                    time_start = (self.time_lookup[bl, tbin - 1] +
                                  0.5*self.interval_lookup[bl, tbin - 1])
                else:
                    # Use the time and interval of the starting row to
                    # establish the start time of the bin
                    time_start = time[rs] - 0.5*interval[rs]

                # Find bin ending time
                if next_row != re:
                    # Use the time and interval of the next row outside
                    # the bin to establish the end time of the bin
                    time_end = time[next_row] - 0.5*interval[next_row]

                    if time_end - time_start > self.time_bin_secs:
                        time_end = time_start + self.time_bin_secs
                else:
                    # Use the time and interval of the ending row
                    # to establish the end time of the bin
                    time_end = time[re] + 0.5*interval[re]

                print(time_start, time_end)

                # Establish the midpoint
                bin_time = 0.5*(time_start + time_end)
                bin_interval = time_end - time_start

            self.time_lookup[bl, tbin] = bin_time
            self.interval_lookup[bl, tbin] = bin_interval
            self.bin_flagged[bl, tbin] = self.bin_count == self.bin_flag_count

            self.tbin += 1

        def execute(self):
            row_lookup = self.row_lookup
            time = self.time
            interval = self.interval
            flag_row = self.flag_row if have_flag_row else None
            bin_lookup = self.bin_lookup

            # Average times over each baseline and construct the
            # bin_lookup and time_lookup arrays
            for bl in range(row_lookup.shape[0]):
                self.start_baseline()

                for t in range(row_lookup.shape[1]):
                    r = row_lookup[bl, t]

                    if r == -1:
                        continue

                    if self.bin_empty:
                        self.start_bin(r, flag_row)
                    elif not self.add_row(r, time, interval, flag_row):
                        # Can't add a new row to this bin, close it
                        # and start a new one
                        self.finalise_bin(bl, r, time, interval)
                        self.start_bin(r, flag_row)

                    # Register the output time bin for this row
                    bin_lookup[bl, t] = self.tbin

                # Close any open bins
                if not self.bin_empty:
                    self.finalise_bin(bl, r, time, interval)

                self.finalise_baseline()

    time = _numba_type(time)
    interval = _numba_type(interval)
    antenna1 = _numba_type(antenna1)
    antenna2 = _numba_type(antenna2)
    time_bin_secs = _numba_type(time_bin_secs)

    spec = [
        ('out_rows', numba.uintp),
        ('rs', numba.intp),
        ('re', numba.intp),
        ('tbin', numba.intp),
        ('bin_count', numba.uintp),
        ('bin_flag_count', numba.uintp),
        ('bl_inv', numba.uintp[:]),
        ('time_inv', numba.uintp[:])]

    spec.extend([
        ('time_lookup', time[:, :]),
        ('interval_lookup', interval[:, :]),
        ('row_lookup', numba.intp[:, :]),
        ('bin_lookup', numba.intp[:, :]),
        ('bin_flagged', numba.bool_[:, :]),
        ('time_bin_secs', time_bin_secs)])

    spec.extend([
        ('time', time[:]),
        ('interval', interval[:])])

    if have_flag_row:
        flag_row = _numba_type(flag_row)
        spec.append(('flag_row', flag_row[:]))

    return jitclass(spec)(Binner)


RowMapOutput = namedtuple("RowMapOutput",
                          ["map", "time", "interval", "flag_row"])


@generated_jit(nopython=True, nogil=True, cache=True)
def row_mapper(time, interval, antenna1, antenna2,
               flag_row=None, time_bin_secs=1):
    """
    Generates a mapping from a high resolution row index to
    a low resolution row index in support of time and channel
    averaging code. The `time` and `interval` columns
    are also respectively averaged and summed
    in the process of creating the mapping and a
    `flag_row` column is returned if one is provided.

    In order to average a chunk of row data, it is necessary to
    group each row (or sample) by baseline and then average
    the time samples present in each baseline in bins of
    `time_bin_secs`.

    Flagged data is handled as follows:

    1. It does not contribute to a bin at all if
       there are other unflagged samples in the bin.
    2. It is the only contribution to a bin if
       all samples in the bin are flagged.

    The algorithm is robust in the presence of missing time and baseline data.

    The algorithm works as follows:

    1. `time`, `interval`, `antenna1` and `antenna2`
    are used to construct a `row_lookup` array of shape `(ubl, utime)`
    mapping a baseline and time to a row of input data.

    2. For each baseline, `time_bin_secs` times are averaged together
    into two separate `time_lookup` arrays of shape `(ubl, utime)`.
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
    time bin in `time_lookup`.

    3. The `time_lookup` array is flattened and argsorted with a stable
    merge sort. As missing values are set to the maximum floating point
    value, this moves valid data to the front and missing data to the back.
    This has the effect of lexicographically sorts the data
    in an ascending `(time, bl)` order

    4. Input rows are then mapped via the `row_lookup`, `bin_lookup`
    and argsorted `time_lookup` arrays to an output row.

    .. code-block:: python

        ret = row_mapper(time, interval,
                         ant1, ant2, flag_row,
                         time_bin_secs=3)

        # Only add a bin's contribution if both input and output
        # are (a) flagged or (b) unflagged
        sel = flag_row == ret.flag_row[ret.map]
        sel_map = ret.map[sel]

        # Recompute time average using row map
        time = np.zeros_like(ret.time)
        ant1_avg = np.empty(time.shape, ant1.dtype)
        ant2_avg = np.empty(time.shape, ant2.dtype)
        counts = np.empty(time.shape, np.uint32)

        # Add time and 1 at map indices to time and counts
        np.add.at(time, sel_map, time[sel])
        np.add.at(counts, sel_map, 1)
        # Normalise
        time /= count

        np.testing.assert_array_equal(time, ret.time)

        # We assign baselines because each input baseline
        # is mapped to the same output baseline
        ant1_avg[sel_map] = ant1[sel]
        ant2_avg[sel_map] = ant2[sel]

    Parameters
    ----------
    time : :class:`numpy.ndarray`
        Time values of shape :code:`(row,)`.
    interval : :class:`numpy.ndarray`
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
    time : :class:`numpy.ndarray`
        Averaged time values of shape :code:`(out_row,)`
    interval : :class:`numpy.ndarray`
        Summed interval values of shape :code:`(out_row,)`
    flag_row : :class:`numpy.ndarray` or None
        Output flag rows of shape :code:`(out_row,)`.
        None if no input flag_row was supplied.

    Raises
    ------
    RowMapperError
        Raised if an illegal condition occurs

    """
    have_flag_row = not is_numba_type_none(flag_row)
    have_time_bin_secs = not is_numba_type_none(time_bin_secs)
    time_bin_secs_type = time_bin_secs if have_time_bin_secs else time.dtype

    JitBinner = binner_factory(time, interval, antenna1, antenna2,
                               flag_row, time_bin_secs_type)

    def impl(time, interval, antenna1, antenna2,
             flag_row=None, time_bin_secs=1):
        # If we don't have time_bin_secs
        # set it to the maximum floating point value,
        # effectively ignoring this limit
        if not have_time_bin_secs:
            time_bin_secs = np.finfo(time.dtype).max

        binner = JitBinner(time, interval, antenna1, antenna2,
                           flag_row, time_bin_secs)
        binner.execute()

        # Flatten the time lookup and argsort it
        flat_time = binner.time_lookup.ravel()
        flat_int = binner.interval_lookup.ravel()
        argsort = np.argsort(flat_time, kind='mergesort')
        inv_argsort = np.empty_like(argsort)

        # Generate lookup from flattened (bl, time) to output row
        for i, a in enumerate(argsort):
            inv_argsort[a] = i

        # Construct the final row map
        row_map = np.empty(time.shape[0], dtype=np.uint32)

        nbl, ntime = binner.row_lookup.shape
        out_rows = binner.out_rows
        bin_lookup = binner.bin_lookup
        bin_flagged = binner.bin_flagged
        bl_inv = binner.bl_inv
        time_inv = binner.time_inv

        # Construct output flag row, if necessary
        out_flag_row = (np.zeros(out_rows, dtype=flag_row.dtype)
                        if have_flag_row else None)

        # foreach input row
        for in_row in range(time.shape[0]):
            # Lookup baseline and time
            bl = bl_inv[in_row]
            t = time_inv[in_row]

            # lookup time bin and output row
            tbin = bin_lookup[bl, t]
            # lookup output row in inv_argsort
            out_row = inv_argsort[bl*ntime + tbin]
            # print(in_row, bl, t, tbin, bl*ntime + tbin, out_row, out_rows)

            if out_row >= out_rows:
                raise RowMapperError("out_row >= out_rows")

            if have_flag_row:
                flagged = bin_flagged[bl, tbin]
                if flag_row[in_row] == 0 and flagged:
                    raise RowMapperError("Unflagged input row contributing "
                                         "to flagged output row. "
                                         "This should never happen!")

                out_flag_row[out_row] = 1 if flagged else 0

            row_map[in_row] = out_row

        time_ret = flat_time[argsort[:out_rows]]
        int_ret = flat_int[argsort[:out_rows]]

        return RowMapOutput(row_map, time_ret, int_ret, out_flag_row)

    return impl


@njit(nogil=True, cache=True)
def channel_mapper(nchan, chan_bin_size=1):
    chan_map = np.empty(nchan, dtype=np.uint32)

    chan_bin = 0
    bin_count = 0

    for c in range(nchan):
        chan_map[c] = chan_bin
        bin_count += 1

        if bin_count == chan_bin_size:
            chan_bin += 1
            bin_count = 0

    if bin_count > 0:
        chan_bin += 1

    return chan_map, chan_bin
