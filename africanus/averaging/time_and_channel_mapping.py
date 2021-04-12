# -*- coding: utf-8 -*-


from collections import namedtuple

import numpy as np
import numba
from numba.experimental import jitclass

from africanus.averaging.support import unique_time, unique_baselines
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


FinaliseOutput = namedtuple("FinaliseOutput",
                            ["tbin", "time", "interval", "flag"])


class Binner:
    def __init__(self, row_start, row_end, time_bin_secs):
        self.rs = row_start
        self.re = row_end
        self.time_bin_secs = time_bin_secs
        self.tbin = 0
        self.bin_count = 0
        self.bin_flag_count = 0

    def reset(self):
        self.__init__(0, 0, self.time_bin_secs)

    @property
    def empty(self):
        return self.bin_count == 0

    def start_bin(self, row, flag_row):
        self.rs = row
        self.re = row
        self.bin_count = 1
        flagged = flag_row is not None and flag_row[row] != 0
        self.bin_flag_count = int(flagged)

    def add_row(self, row, auto_corr, time, interval, flag_row):
        rs = self.rs
        re = self.re

        if re == row:
            raise ValueError("start_bin should be called to start a bin "
                             "before add_row is called.")

        dt = (time[row] + 0.5*interval[row]) - (time[rs] - 0.5*interval[rs])

        if dt > self.time_bin_secs:
            return False
        else:
            self.re = row
            self.bin_count += 1
            flagged = flag_row is not None and flag_row[row] != 0
            self.bin_flag_count += int(flagged)

        return True

    def finalise_bin(self, time, interval):
        rs = self.rs
        re = self.re

        if rs == re:
            bin_time = time[rs]
            bin_interval = interval[rs]
        else:
            dt = time[re] - time[rs]
            bin_time = 0.5*(time[re] + time[rs])
            bin_interval = 0.5*interval[rs] + 0.5*interval[re] + dt

        out = FinaliseOutput(self.tbin, bin_time, bin_interval,
                             self.bin_count == self.bin_flag_count)

        self.tbin += 1
        return out


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

    output_flag_row = output_factory(have_flag_row)
    set_flag_row = set_flag_row_factory(have_flag_row)

    have_time_bin_secs = not is_numba_type_none(time_bin_secs)
    time_bin_secs_type = time_bin_secs if have_time_bin_secs else time.dtype

    spec = [
        ('tbin', numba.uintp),
        ('bin_count', numba.uintp),
        ('bin_flag_count', numba.uintp),
        ('rs', numba.uintp),
        ('re', numba.uintp),
        ('time_bin_secs', time_bin_secs_type)]

    JitBinner = jitclass(spec)(Binner)

    def impl(time, interval, antenna1, antenna2,
             flag_row=None, time_bin_secs=1):
        ubl, _, bl_inv, _ = unique_baselines(antenna1, antenna2)
        utime, _, time_inv, _ = unique_time(time)

        nbl = ubl.shape[0]
        ntime = utime.shape[0]

        sentinel = np.finfo(time.dtype).max
        out_rows = numba.uint32(0)

        scratch = np.full(3*nbl*ntime, -1, dtype=np.int32)
        row_lookup = scratch[:nbl*ntime].reshape(nbl, ntime)
        bin_lookup = scratch[nbl*ntime:2*nbl*ntime].reshape(nbl, ntime)
        inv_argsort = scratch[2*nbl*ntime:]
        time_lookup = np.zeros((nbl, ntime), dtype=time.dtype)
        interval_lookup = np.zeros((nbl, ntime), dtype=interval.dtype)

        # Is the entire bin flagged?
        bin_flagged = np.zeros((nbl, ntime), dtype=np.bool_)

        # Create a mapping from the full bl x time resolution back
        # to the original input rows
        for r in range(time.shape[0]):
            bl = bl_inv[r]
            t = time_inv[r]

            if row_lookup[bl, t] == -1:
                row_lookup[bl, t] = r
            else:
                raise ValueError("Duplicate (TIME, ANTENNA1, ANTENNA2) "
                                 "combinations were discovered in the input "
                                 "data. This is usually caused by not "
                                 "partitioning your data sufficiently "
                                 "by indexing columns, DATA_DESC_ID "
                                 "and SCAN_NUMBER in particular.")

        # If we don't have time_bin_secs
        # set it to the maximum floating point value,
        # effectively ignoring this limit
        if not have_time_bin_secs:
            time_bin_secs = np.finfo(time.dtype).max

        binner = JitBinner(0, 0, time_bin_secs)

        # Average times over each baseline and construct the
        # bin_lookup and time_lookup arrays
        for bl in range(ubl.shape[0]):
            binner.reset()

            # Auto-correlated baseline
            auto_corr = ubl[bl, 0] == ubl[bl, 1]

            for t in range(utime.shape[0]):
                # Lookup input row
                r = row_lookup[bl, t]

                # Ignore if not present
                if r == -1:
                    continue

                if binner.empty:
                    binner.start_bin(r, flag_row)
                elif not binner.add_row(r, auto_corr, time,
                                        interval, flag_row):
                    f = binner.finalise_bin(time, interval)
                    time_lookup[bl, f.tbin] = f.time
                    interval_lookup[bl, f.tbin] = f.interval
                    bin_flagged[bl, f.tbin] = f.flag
                    binner.start_bin(r, flag_row)

                # Record the output bin associated with the row
                bin_lookup[bl, t] = binner.tbin

            # Normalise the last bin if it has entries in it
            if not binner.empty:
                f = binner.finalise_bin(time, interval)
                time_lookup[bl, f.tbin] = f.time
                interval_lookup[bl, f.tbin] = f.interval
                bin_flagged[bl, f.tbin] = f.flag

            # Add this baseline's number of bins to the output rows
            out_rows += binner.tbin

            # Set any remaining bins to sentinel value and unflagged
            for tbin in range(binner.tbin, ntime):
                time_lookup[bl, tbin] = sentinel
                bin_flagged[bl, tbin] = False

        # Flatten the time lookup and argsort it
        flat_time = time_lookup.ravel()
        flat_int = interval_lookup.ravel()
        argsort = np.argsort(flat_time, kind='mergesort')

        # Generate lookup from flattened (bl, time) to output row
        for i, a in enumerate(argsort):
            inv_argsort[a] = i

        # Construct the final row map
        row_map = np.empty(time.shape[0], dtype=np.uint32)

        # Construct output flag row, if necessary
        out_flag_row = output_flag_row(out_rows, flag_row)

        # foreach input row
        for in_row in range(time.shape[0]):
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
        int_ret = flat_int[argsort[:out_rows]]

        return RowMapOutput(row_map, time_ret, int_ret, out_flag_row)

    return impl


@jit(nopython=True, nogil=True, cache=True)
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
