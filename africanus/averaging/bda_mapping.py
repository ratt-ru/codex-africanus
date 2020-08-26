# -*- coding: utf-8 -*-

from collections import namedtuple

import numpy as np
import numba
from numba.experimental import jitclass
import numba.types

from africanus.constants import c as lightspeed
from africanus.util.numba import generated_jit, njit
from africanus.averaging.support import unique_time, unique_baselines


class RowMapperError(Exception):
    pass


_SERIES_COEFFS = (1./40, 107./67200, 3197./24192000, 49513./3973939200)


@njit(nogil=True, cache=True, inline='always')
def inv_sinc(sinc_x, tol=1e-12):
    # Initial guess from reversion of Taylor series
    # https://math.stackexchange.com/questions/3189307/inverse-of-frac-sinxx
    x = t_pow = np.sqrt(6*(1 - sinc_x))
    t_squared = t_pow*t_pow

    for coeff in numba.literal_unroll(_SERIES_COEFFS):
        t_pow *= t_squared
        x += coeff * t_pow

    # Use Newton Raphson to go the rest of the way
    # https://www.wolframalpha.com/input/?i=simplify+%28sinc%5Bx%5D+-+c%29+%2F+D%5Bsinc%5Bx%5D%2Cx%5D
    while True:
        # evaluate delta between this iteration sinc(x) and original
        sinx = np.sin(x)
        𝞓sinc_x = (1.0 if x == 0.0 else sinx/x) - sinc_x

        # Stop if converged
        if np.abs(𝞓sinc_x) < tol:
            break

        # Next iteration
        x -= (x*x * 𝞓sinc_x) / (x*np.cos(x) - sinx)

    return x


@njit(nogil=True, cache=True, inline='always')
def factors(n):
    result = []
    i = 1

    while i*i < n:
        quot, rem = divmod(n, i)

        if rem == 0:
            result.append(i)

            if quot != i:
                result.append(quot)

        i += 1

    return np.unique(np.array(result))


@njit(nogil=True, cache=True, inline='always')
def max_chan_width(ref_freq, fractional_bandwidth):
    """
    Derive max_𝞓𝝼, the maximum change in bandwidth
    before decorrelation occurs in frequency

    Fractional Bandwidth is defined by
    https://en.wikipedia.org/wiki/Bandwidth_(signal_processing)
    for Wideband Antennas as:
      (1) 𝞓𝝼/𝝼 = fb = (fh - fl) / (fh + fl)
    where fh and fl are the high and low frequencies
    of the band.
    We set fh = ref_freq + 𝞓𝝼/2, fl = ref_freq - 𝞓𝝼/2
    Then, simplifying (1), 𝞓𝝼 = 2 * ref_freq * fb
    """
    return 2 * ref_freq * fractional_bandwidth


FinaliseOutput = namedtuple("FinaliseOutput",
                            ["tbin", "time", "interval",
                             "nchan", "flag"])


class Binner(object):
    def __init__(self, row_start, row_end,
                 max_lm,
                 ref_freq,
                 decorrelation):
        # Index of the time bin to which all rows in the bin will contribute
        self.tbin = 0
        # Number of rows in the bin
        self.bin_count = 0
        # Number of flagged rows in the bin
        self.bin_flag_count = 0
        # Time sum
        self.time_sum = 0.0
        # Interval sum
        self.interval_sum = 0.0
        # Starting row of the bin
        self.rs = row_start
        # Ending row of the bin
        self.re = row_end
        # Sinc of half the baseline speed
        self.bin_sinc_half_Δψ = 0.0

        # Quantities cached to make Binner.method arguments smaller
        self.ref_freq = ref_freq
        self.max_lm = max_lm
        n = -1.0 if max_lm > 1.0 else np.sqrt(1.0 - max_lm**2) - 1.0
        self.n_max = np.abs(n)
        self.decorrelation = decorrelation

    def reset(self):
        self.__init__(0, 0, self.max_lm, self.ref_freq, self.decorrelation)

    def start_bin(self, row, time, interval, flag_row):
        """
        Starts a new bin
        """
        self.rs = row
        self.re = row
        self.bin_count = 1
        self.time_sum = time[row]
        self.interval_sum = interval[row]
        self.bin_flag_count = (1 if flag_row is not None and flag_row[row] != 0
                               else 0)

    def add_row(self, row, auto_corr, time, interval, uvw, flag_row):
        """
        Attempts to add ``row`` to the current bin.

        Returns
        -------
        success : bool
            True if the decorrelation tolerance was not exceeded
            and the row was added to the bin.
        """
        rs = self.rs
        re = self.re

        if re == row:
            raise ValueError("start_bin should be called to start a bin "
                             "before add_row is called.")

        if auto_corr:
            # Fast path for auto-correlated baseline.
            # By definition, uvw == (0, 0, 0) for these samples
            self.re = row
            self.bin_sinc_half_Δψ = self.decorrelation
            self.bin_count += 1
            self.time_sum += time[row]
            self.interval_sum += interval[row]

            if flag_row is not None and flag_row[row] != 0:
                self.bin_flag_count += 1

            return True

        # Evaluate the degree of decorrelation
        # the sample would add to existing bin
        dt = (time[row] + (interval[row] / 2.0) -
              (time[rs] - interval[rs] / 2.0))
        du = uvw[row, 0] - uvw[rs, 0]
        dv = uvw[row, 1] - uvw[rs, 1]

        du_dt = du / dt
        dv_dt = dv / dt
        # Derive phase difference in time
        # from Equation (36) in Atemkeng
        # leaving out the factor of two
        # which is divided out when computing the sinc
        half_𝞓𝞇 = np.pi * np.sqrt(du_dt**2 + dv_dt**2) * self.max_lm
        sinc_half_𝞓𝞇 = 1.0 if half_𝞓𝞇 == 0.0 else np.sin(half_𝞓𝞇) / half_𝞓𝞇

        # Do not add the row to the bin as it
        # would exceed the decorrelation tolerance
        if sinc_half_𝞓𝞇 <= self.decorrelation:
            return False

        # Add the row by making it the end of the bin
        # and keep a record of the sinc_half_𝞓𝞇
        self.re = row
        self.bin_sinc_half_Δψ = sinc_half_𝞓𝞇
        self.bin_count += 1
        self.time_sum += time[row]
        self.interval_sum += interval[row]

        if flag_row is not None and flag_row[row] != 0:
            self.bin_flag_count += 1

        return True

    @property
    def empty(self):
        return self.bin_count == 0

    def finalise_bin(self, auto_corr, uvw, nchan_factors,
                     chan_width, chan_freq):
        """ Finalise the contents of this bin """
        if self.bin_count == 0:
            raise ValueError("Attempted to finalise empty bin")

        rs = self.rs
        re = self.re

        # Calculate the maximum change in frequency for the bin,
        # given the change in phase
        if auto_corr:
            # Auto-correlated baseline, average all channels
            # everything down to a single value
            nchan = 1
        else:
            # Central UVW coordinate of the bin
            cu = (uvw[rs, 0] + uvw[re, 0]) / 2
            cv = (uvw[rs, 1] + uvw[re, 1]) / 2
            cw = (uvw[rs, 2] + uvw[re, 2]) / 2

            cuv = np.sqrt(cu**2 + cv**2)

            max_abs_dist = np.sqrt(np.abs(cuv)*np.abs(self.max_lm) +
                                   np.abs(cw)*np.abs(self.n_max))

            if max_abs_dist == 0.0:
                raise ValueError("max_abs_dist == 0.0")

            # Given
            #   (1) acceptable decorrelation
            #   (2) change in (phase) baseline speed
            # derive the frequency phase difference
            # from Equation (40) in Atemkeng
            # If there's a single sample (rs == re)
            # we can't meaningfully calculate baseline speed.
            # In this case frequency phase difference
            # just becomes the decorrelation factor
            half_sinc_𝞓𝞍 = (self.decorrelation if rs == re else
                            self.decorrelation / self.bin_sinc_half_Δψ)
            half_𝞓𝞍 = inv_sinc(half_sinc_𝞓𝞍)

            max_𝞓𝝼 = (half_𝞓𝞍 / np.pi) * (lightspeed / max_abs_dist)
            nchan = max(int(1), int(chan_width.sum() / max_𝞓𝝼))

            # Now find the next highest integer factorisation
            # of the input number of channels
            s = np.searchsorted(nchan_factors, nchan, side='left')
            nchan = nchan_factors[min(nchan_factors.shape[0] - 1, s)]

            # print(self.bin_count, half_sinc_𝞓𝞍, max_𝞓𝝼, nchan1, nchan)

        # Finalise bin values for return
        out = FinaliseOutput(self.tbin,
                             self.time_sum / self.bin_count,
                             self.interval_sum,
                             nchan,
                             self.bin_count == self.bin_flag_count)

        self.tbin += 1

        return out


RowMapOutput = namedtuple("RowMapOutput",
                          ["map", "offsets", "num_chan", "decorr_chan_width",
                           "time", "interval", "chan_width", "flag_row"])


@generated_jit(nopython=True, nogil=True, cache=True)
def atemkeng_mapper(time, interval, ant1, ant2, uvw,
                    chan_width, chan_freq,
                    ref_freq, max_uvw_dist,
                    flag_row=None, max_fov=3.0,
                    decorrelation=0.98, min_nchan=1):

    Omitted = numba.types.misc.Omitted

    decorr_type = (numba.typeof(decorrelation.value)
                   if isinstance(decorrelation, Omitted)
                   else decorrelation)

    fov_type = (numba.typeof(max_fov.value)
                if isinstance(max_fov, Omitted)
                else max_fov)

    ref_freq_dtype = ref_freq

    spec = [
        ('tbin', numba.uintp),
        ('bin_count', numba.uintp),
        ('bin_flag_count', numba.uintp),
        ('time_sum', time.dtype),
        ('interval_sum', interval.dtype),
        ('rs', numba.uintp),
        ('re', numba.uintp),
        ('bin_sinc_half_Δψ', uvw.dtype),
        ('max_lm', fov_type),
        ('n_max', fov_type),
        ('ref_freq', ref_freq_dtype),
        ('decorrelation', decorr_type),
        ('max_uvw_dist', max_uvw_dist)]

    JitBinner = jitclass(spec)(Binner)

    def impl(time, interval, ant1, ant2, uvw,
             chan_width, chan_freq,
             ref_freq, max_uvw_dist,
             flag_row=None, max_fov=3.0,
             decorrelation=0.98, min_nchan=1):
        # 𝞓 𝝿 𝞇 𝞍 𝝼

        if decorrelation < 0.0 or decorrelation > 1.0:
            raise ValueError("0.0 <= decorrelation <= 1.0 must hold")

        if max_fov <= 0.0 or max_fov > 90.0:
            raise ValueError("0.0 < max_fov <= 90.0 must hold")

        max_lm = np.sin(np.deg2rad(max_fov))

        ubl, _, bl_inv, _ = unique_baselines(ant1, ant2)
        utime, _, time_inv, _ = unique_time(time)

        nrow = time.shape[0]
        ntime = utime.shape[0]
        nbl = ubl.shape[0]
        nchan = chan_width.shape[0]
        nchan_factors = factors(nchan)
        bandwidth = chan_width.sum()

        if min_nchan is None:
            min_nchan = 1
        else:
            s = np.searchsorted(nchan_factors, min_nchan, side='left')
            min_nchan = max(min_nchan, nchan_factors[s])

        if nchan == 0:
            raise ValueError("zero channels")

        binner = JitBinner(0, 0, max_lm, ref_freq, decorrelation)

        # Create the row lookup
        row_lookup = np.full((nbl, ntime), -1, dtype=np.int32)
        bin_lookup = np.full((nbl, ntime), -1, dtype=np.int32)
        bin_chan_width = np.full((nbl, ntime), 0.0, dtype=ref_freq_dtype)
        sentinel = np.finfo(time.dtype).max
        time_lookup = np.full((nbl, ntime), sentinel, dtype=time.dtype)
        interval_lookup = np.full((nbl, ntime), sentinel, dtype=interval.dtype)
        # Is the entire bin flagged?
        bin_flagged = np.zeros((nbl, ntime), dtype=np.bool_)
        bin_chan_map = np.empty((nbl, ntime, nchan), dtype=np.int32)

        out_rows = 0
        nr_of_time_bins = 0
        out_row_chans = 0

        def update_lookups(finalised, bl):
            """
            Closure which updates lookups for a baseline,
            given a binner's finalisation data
            """
            # NOTE(sjperkins) Why do scalars need this, but not arrays?
            nonlocal out_rows
            nonlocal out_row_chans
            nonlocal min_nchan

            tbin = finalised.tbin

            time_lookup[bl, tbin] = finalised.time
            interval_lookup[bl, tbin] = finalised.interval
            bin_flagged[bl, tbin] = finalised.flag
            nchan = max(finalised.nchan, min_nchan)
            bin_nchan = chan_width.shape[0] // nchan
            bin_chan_width[bl, tbin] = bandwidth / finalised.nchan

            # Construct the channel map
            for c in range(chan_width.shape[0]):
                bin_chan_map[bl, tbin, c] = c // bin_nchan

            out_rows += 1
            out_row_chans += nchan

        for r in range(nrow):
            t = time_inv[r]
            bl = bl_inv[r]

            if row_lookup[bl, t] != -1:
                raise ValueError("Duplicate (TIME, ANTENNA1, ANTENNA2)")

            row_lookup[bl, t] = r

        for bl in range(nbl):
            # Reset the binner for this baseline
            binner.reset()

            # Auto-correlated baseline
            auto_corr = ubl[bl, 0] == ubl[bl, 1]

            for t in range(ntime):
                # Lookup row, continue if non-existent
                r = row_lookup[bl, t]

                if r == -1:
                    continue

                # Start a new bin
                if binner.empty:
                    binner.start_bin(r, time, interval, flag_row)
                # Try add the row to the bin
                # If this fails, finalise the current bin and start a new one
                elif not binner.add_row(r, auto_corr,
                                        time, interval,
                                        uvw, flag_row):
                    f = binner.finalise_bin(auto_corr, uvw,
                                            nchan_factors,
                                            chan_width, chan_freq)
                    update_lookups(f, bl)
                    # Post-finalisation, the bin is empty, start a new bin
                    binner.start_bin(r, time, interval, flag_row)

                # Record the time bin associated with this row
                bin_lookup[bl, t] = binner.tbin

            # Finalise any remaining data in the bin
            if not binner.empty:
                f = binner.finalise_bin(auto_corr, uvw,
                                        nchan_factors,
                                        chan_width, chan_freq)
                update_lookups(f, bl)

            nr_of_time_bins += binner.tbin

            # Mark remaining bins as unoccupied and unflagged
            for tbin in range(binner.tbin, ntime):
                time_lookup[bl, tbin] = sentinel
                bin_flagged[bl, tbin] = False

        assert out_rows == nr_of_time_bins

        # Flatten the time lookup and argsort it
        flat_time = time_lookup.ravel()
        argsort = np.argsort(flat_time, kind='mergesort')
        inv_argsort = np.empty_like(argsort)

        # Generate lookup from flattened (bl, time) to output row
        for i, a in enumerate(argsort):
            inv_argsort[a] = i

        # Generate row offsets
        fbin_chan_map = bin_chan_map.reshape((-1, nchan))
        offsets = np.empty(out_rows, dtype=np.uint32)
        out_chans = np.empty(out_rows, dtype=np.uint32)
        decorr_chan_width = np.empty(out_rows, dtype=chan_width.dtype)

        # NOTE(sjperkins)
        # This: out_rows > 0
        # does not work here for some strange (numba reason?)
        if offsets.shape[0] > 0:
            offsets[0] = 0
            out_chans[0] = fbin_chan_map[argsort[0]].max() + 1

            for r in range(1, offsets.shape[0]):
                prev_bin_chans = out_chans[r - 1]
                offsets[r] = offsets[r - 1] + prev_bin_chans
                out_chans[r] = fbin_chan_map[argsort[r]].max() + 1

        # Construct the final row map
        row_chan_map = np.full((nrow, nchan), -1, dtype=np.int32)
        time_ret = np.full(out_row_chans, -1, dtype=time.dtype)
        int_ret = np.full(out_row_chans, -1, dtype=interval.dtype)
        chan_width_ret = np.full(out_row_chans, -1, dtype=chan_width.dtype)

        # Construct output flag row, if necessary
        out_flag_row = (None if flag_row is None else
                        np.empty(out_row_chans, dtype=flag_row.dtype))

        # foreach input row
        for in_row in range(time.shape[0]):
            # Lookup baseline and time
            bl = bl_inv[in_row]
            t = time_inv[in_row]

            # lookup time bin and output row in inv_argsort
            tbin = bin_lookup[bl, t]
            bin_time = time_lookup[bl, tbin]
            bin_interval = interval_lookup[bl, tbin]
            flagged = bin_flagged[bl, tbin]
            out_row = inv_argsort[bl*ntime + tbin]

            decorr_chan_width[out_row] = bin_chan_width[bl, tbin]

            # Should never happen, but check
            if out_row >= out_rows:
                raise RowMapperError("out_row >= out_rows")

            # Handle output row flagging
            if flag_row is not None and flag_row[in_row] == 0 and flagged:
                raise RowMapperError("Unflagged input row "
                                     "contributing to "
                                     "flagged output row. "
                                     "This should never happen!")

            # Set up the row channel map, populate
            # time, interval and chan_width
            for c in range(nchan):
                out_offset = offsets[out_row] + bin_chan_map[bl, tbin, c]

                # Should never happen, but check
                if out_offset >= out_row_chans:
                    raise RowMapperError("out_offset >= out_row_chans")

                # Set the output row for this input row and channel
                row_chan_map[in_row, c] = out_offset

                # Broadcast the time and interval to the output row
                time_ret[out_offset] = bin_time
                int_ret[out_offset] = bin_interval

                # Add channel contribution for each row
                chan_width_ret[out_offset] += chan_width[c]

                if flag_row is not None:
                    out_flag_row[out_offset] = 1 if flagged else 0

        return RowMapOutput(row_chan_map, offsets, out_chans,
                            decorr_chan_width,
                            time_ret, int_ret,
                            chan_width_ret, out_flag_row)

    return impl
