# -*- coding: utf-8 -*-

import numpy as np
import numba

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

from africanus.constants import c as lightspeed
from africanus.util.numba import generated_jit, njit
from africanus.averaging.support import unique_time, unique_baselines

class RowMapperError(Exception):
    pass

    # if isinstance(lm_max, np.ndarray):
    #     if not lm_max.shape == (2,):
    #         raise ValueError("lm_max must have shape 2 if an ndarray")

    #     l, m = lm_max
    # elif isinstance(lm_max, (tuple, list)):
    #     if not len(lm_max) == 2:
    #         raise ValueError("lm_max must have length 2 if tuple/list")

    #     l, m = lm_max
    # elif isinstance(lm_max, (int, float)):
    #     l = m = np.sqrt(lm_max / 2)
    # else:
    #     raise TypeError("lm_max must a single float, or a "
    #                     "tuple/list/ndarray of 2 elements")




def duv_dt(utime, ubl, ants, chan_freq, phase_dir):
    # Unique ant1 and ant2
    uant1 = ubl[:, 0]
    uant2 = ubl[:, 1]

    # Dimensions
    ntime = utime.shape[0]
    nbl = ubl.shape[0]
    nchan = chan_freq.shape[0]

    ast_centre = EarthLocation.from_geocentric(ants[:, 0].mean(),
                                               ants[:, 1].mean(),
                                               ants[:, 2].mean(),
                                               unit="m")
    lon, lat, alt = ast_centre.to_geodetic()
    ast_time = Time(utime / 86400.00, format='mjd', scale='ut1')
    ast_phase_dir = SkyCoord(ra=phase_dir[0], dec=phase_dir[1], unit='rad')

    # Get hour angle and dec, convert to radians
    ha = ast_time.sidereal_time("apparent", lon) - ast_phase_dir.ra
    ha = ha.to('rad').value
    dec = ast_phase_dir.dec.to('rad').value

    # Numeric derivative of the hour angle with respect to time
    # http://kitchingroup.cheme.cmu.edu/blog/2013/02/27/Numeric-derivatives-by-differences/
    𝞓ha𝞓t = np.empty_like(ha)
    t = utime
    𝞓ha𝞓t[0] = (ha[0] - ha[1]) / (t[0] - t[1])
    𝞓ha𝞓t[1:-1] = (ha[2:] - ha[:-2]) / (t[2:] - t[:-2])
    𝞓ha𝞓t[-1] = (ha[-1] - ha[-2]) / (t[-1] - t[-2])

    # Baseline antenna position difference
    Lx = (ants[uant1, 0] - ants[uant2, 0])[:, None, None]
    Ly = (ants[uant1, 1] - ants[uant2, 1])[:, None, None]
    Lz = (ants[uant2, 2] - ants[uant2, 2])[:, None, None]

    # Prepare the result array
    dtype = np.result_type(Lx, ha, chan_freq)
    𝞓uv𝞓t = np.empty((nbl, ntime, nchan, 2), dtype)

    # Synthesis and Imaging 18-33 and 18-34
    spatial_freq = chan_freq[None, None, :] / lightspeed
    ha = ha[None, :, None]
    𝞓uv𝞓t[..., 0] = spatial_freq * (Lx*np.cos(ha) - Ly*np.sin(ha))
    𝞓uv𝞓t[..., 1] = spatial_freq * (Lx*np.sin(dec)*np.sin(ha) +
                                     Ly*np.sin(dec)*np.cos(ha))
    𝞓uv𝞓t *= 𝞓ha𝞓t[None, :, None, None]
    return 𝞓uv𝞓t

def row_mapper(time, uvw, ants, phase_dir, ref_freq, lm_max):
    """
    Parameters
    ----------

    Returns
    -------
    """
    time = np.unique(time)  # Remove duplicate times

    if isinstance(lm_max, np.ndarray):
        if not lm_max.shape == (2,):
            raise ValueError("lm_max must have shape 2 if an ndarray")

        l, m = lm_max
    elif isinstance(lm_max, (tuple, list)):
        if not len(lm_max) == 2:
            raise ValueError("lm_max must have length 2 if tuple/list")

        l, m = lm_max
    elif isinstance(lm_max, (int, float)):
        l = m = np.sqrt(lm_max / 2)
    else:
        raise TypeError("lm_max must a single float, or a "
                        "tuple/list/ndarray of 2 elements")

    ant1, ant2 = (a.astype(np.int32) for a in np.triu_indices(ants.shape[0], 1))
    ntime = time.shape[0]
    nbl = ant1.shape[0]

    ast_centre = EarthLocation.from_geocentric(ants[:, 0].mean(),
                                               ants[:, 1].mean(),
                                               ants[:, 2].mean(),
                                               unit="m")
    lon, lat, alt = ast_centre.to_geodetic()
    ast_time = Time(time / 86400.00, format='mjd', scale='ut1')
    ast_phase_dir = SkyCoord(ra=phase_dir[0], dec=phase_dir[1], unit='rad')

    # Get hour angle and dec, convert to radians
    ha = ast_time.sidereal_time("apparent", lon) - ast_phase_dir.ra
    ha = ha.to('rad').value
    dec = ast_phase_dir.dec.to('rad').value

    # Numeric derivate of the hour angle with respect to time
    # http://kitchingroup.cheme.cmu.edu/blog/2013/02/27/Numeric-derivatives-by-differences/
    𝞓ha𝞓t = np.empty_like(ha)
    t = time
    𝞓ha𝞓t[0] = (ha[0] - ha[1]) / (t[0] - t[1])
    𝞓ha𝞓t[1:-1] = (ha[2:] - ha[:-2]) / (t[2:] - t[:-2])
    𝞓ha𝞓t[-1] = (ha[-1] - ha[-2]) / (t[-1] - t[-2])

    # Baseline antenna position difference
    Lx = (ants[ant1, 0] - ants[ant2, 0])[:, None]
    Ly = (ants[ant1, 1] - ants[ant2, 1])[:, None]
    Lz = (ants[ant2, 2] - ants[ant2, 2])[:, None]

    # Synthesis and Imaging 18-33 and 18-34
    spatial_freq = ref_freq / lightspeed
    ha = ha[None, :]
    𝞓ha𝞓t = 𝞓ha𝞓t[None, :]
    𝞓u𝞓t = spatial_freq * (Lx*np.cos(ha) - Ly*np.sin(ha))
    𝞓v𝞓t = spatial_freq * (Lx*np.sin(dec)*np.sin(ha) + Ly*np.sin(dec)*np.cos(ha))
    𝞓u𝞓t *= 𝞓ha𝞓t
    𝞓v𝞓t *= 𝞓ha𝞓t

    # (bl, time)
    assert 𝞓u𝞓t.shape == (Lx.shape[0], time.shape[0])

    # 𝞓u𝞓t = 𝞓u𝞓t.ravel()
    # 𝞓v𝞓t = 𝞓v𝞓t.ravel()

    # Synthesis and Imaging 18-31. Reduction in Amplitude
    𝞓𝞍𝞓t = 𝞓u𝞓t*l + 𝞓v𝞓t*m

    v = 𝞓𝞍𝞓t* np.pi
    v[v == 0] = 1.0
    R = np.sin(v) / v

    # 𝞓𝞍𝞓t = 2 * np.pi * 𝞓𝞍𝞓t[:, :, None] * chan_freq[None, None, :] / lightspeed

    return R

_SERIES_COEFFS = (1./40, 107./67200, 3197./24192000, 49513./3973939200)

@njit(nogil=True, cache=True, inline='always')
def inv_sinc(sinc_x, tol=1e-12):
    # Initial guess from reversion of Taylor series
    # https://math.stackexchange.com/questions/3189307/inverse-of-frac-sinxx
    x = t_pow = np.sqrt(6*(1 - sinc_x))
    t_squared = t_pow*t_pow

    for coeff in _SERIES_COEFFS:
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
        x -= (x*x*𝞓sinc_x) / (x*np.cos(x) - sinx)

    return x

def blah(x):
    eps = 1.0
    while np.abs(eps) > 1e-12:
        sinx = np.sin(x)

        # For very small x, the denominator in the gauss newtown
        # evaluates to zero. In any case
        if x == sinx:
            print("Failure")
            break

        cosx = np.cos(x)

        print(x, sinx/x, np.abs(eps), sinx, cosx, (x*x - 2) * sinx + 2*x*cosx)
        # https://www.wolframalpha.com/input/?i=simplify+D%5BSinc%5Bx%5D%2C+x%5D%2FD%5BSinc%5Bx%5D%2C+%7Bx%2C+2%7D%5D
        eps = - (x*(sinx - x*cosx))/((x*x - 2)*sinx + 2*x*cosx)
        x += eps

    final_sinc_x = 1.0 if x == 0.0 else sinx / x

    print(x, sinc_x, final_sinc_x)

    return x


class Binner(object):
    def __init__(self, row_start, row_end,
                 ref_freq, l, m, n_max,
                 decorrelation):
        self.tbin = 0
        self.bin_count = 0
        self.bin_flag_count = 0
        self.rs = row_start
        self.re = row_end
        self.bin_sinc_Δψ = 0.0
        self.ref_freq = ref_freq
        self.l = l
        self.m = m
        self.n_max = n_max
        self.decorrelation = decorrelation

    def reset(self):
        self.__init__(0, 0, self.ref_freq,
                      self.l, self.m, self.n_max,
                      self.decorrelation)

    def start_bin(self, row):
        self.rs = row
        self.re = row
        self.bin_count = 1

    def add_row(self, row, time, interval, uvw):
        rs = self.rs
        re = self.re
        # Evaluate the degree of decorrelation
        # the sample would add to existing bin
        dt = (time[row] + (interval[row] / 2.0) -
            (time[rs] - interval[rs] / 2.0))
        du = uvw[re, 0] - uvw[row, 0]
        dv = uvw[re, 1] - uvw[row, 1]
        dw = uvw[re, 2] - uvw[row, 2]

        du_dt = self.l * du / dt
        dv_dt = self.m * dv / dt

        # Derive phase difference in time
        # from Equation (33) in Atemkeng
        # without factor of 2
        𝞓𝞇 = np.pi * (du_dt + dv_dt)
        sinc_𝞓𝞇 = 1.0 if 𝞓𝞇 == 0.0 else np.sin(𝞓𝞇) / 𝞓𝞇

        # We're not decorrelated at this point,
        # but keep a record of the sinc_𝞓𝞇
        # and the end of the bin
        if sinc_𝞓𝞇 > self.decorrelation:
            self.bin_sinc_Δψ = sinc_𝞓𝞇
            self.re = row
            self.bin_count += 1
            return False

        # Adding row to the bin would decorrelate it
        return True

    def finalise_bin(self, uvw, chan_freq, chan_width):
        # Contents of the bin exceed decorrelation tolerance
        # Finalise it and start a new one

        rs = self.rs
        re = self.re

        # Handle special case of bin containing a single sample.
        # Change in baseline speed 𝞓𝞇 == 0
        if self.bin_count == 1:
            if rs != re:
                raise ValueError("single row in bin, but "
                                 "start row != end row")

            du = uvw[rs, 0]
            dv = uvw[rs, 1]
            dw = uvw[rs, 2]
            bin_sinc_Δψ =  1.0
        else:
            # duvw between start and end row
            du = uvw[rs, 0] - uvw[re, 0]
            dv = uvw[rs, 1] - uvw[re, 1]
            dw = uvw[rs, 2] - uvw[re, 2]
            bin_sinc_Δψ = self.bin_sinc_Δψ

        # Derive fractional bandwidth 𝞓𝝼/𝝼
        # from Equation (44) in Atemkeng
        max_abs_dist = np.sqrt(np.abs(du)*np.abs(self.l) +
                               np.abs(dv)*np.abs(self.m) +
                               np.abs(dw)*np.abs(self.n_max))

        if max_abs_dist == 0.0:
            raise ValueError("max_abs_dist == 0.0")

        # Given
        #   (1) acceptable decorrelation
        #   (2) change in baseline speed
        # derive the frequency phase difference
        # from Equation (35) in Atemkeng
        sinc_𝞓𝞍 = self.decorrelation / bin_sinc_Δψ
        𝞓𝞍 = inv_sinc(sinc_𝞓𝞍)
        fractional_bandwidth = 𝞓𝞍 / max_abs_dist

        # Derive max_𝞓𝝼, the maximum change in bandwidth
        # before decorrelation occurs in frequency
        #
        # Fractional Bandwidth is defined by
        # https://en.wikipedia.org/wiki/Bandwidth_(signal_processing)
        # for Wideband Antennas as:
        #   (1) 𝞓𝝼/𝝼 = fb = (fh - fl) / (fh + fl)
        # where fh and fl are the high and low frequencies
        # of the band.
        # We set fh = ref_freq + 𝞓𝝼/2, fl = ref_freq - 𝞓𝝼/2
        # Then, simplifying (1), 𝞓𝝼 = 2 * ref_freq * fb
        max_𝞓𝝼 = 2 * self.ref_freq * fractional_bandwidth

        bin_freq_low = chan_freq[0] - chan_width[0] / 2
        bin_chan_low = 0

        chan_bins = 0

        for c in range(1, chan_freq.shape[0]):
            # Bin bandwidth
            bin_𝞓𝝼 = chan_freq[c] + chan_width[c] / 2 - bin_freq_low

            # Exceeds, start new channel bin
            if bin_𝞓𝝼 > max_𝞓𝝼:
                bin_chan_low = c
                bin_freq_low = chan_freq[c] - chan_width[c] / 2
                chan_bins += 1

        if bin_𝞓𝝼 > max_𝞓𝝼:
            chan_bins += 1
            bin_freq_low = chan_freq[c] - chan_width[c] / 2

        self.tbin += 1
        self.bin_count = 0
        self.bin_flag_count = 0

from numba.experimental import jitclass
import numba.types

@generated_jit(nopython=True, nogil=True, cache=True)
def atemkeng_mapper(time, interval, ant1, ant2, uvw,
                    ref_freq, chan_freq, chan_width,
                    lm_max=1.0, decorrelation=0.98):

    Omitted = numba.types.misc.Omitted

    decorr_type = (numba.typeof(decorrelation.value)
                   if isinstance(decorrelation, Omitted)
                   else decorrelation)

    lm_type = (numba.typeof(lm_max.value)
               if isinstance(lm_max, Omitted)
               else lm_max)

    spec = [
        ('tbin', numba.uintp),
        ('bin_count', numba.uintp),
        ('bin_flag_count', numba.uintp),
        ('rs', numba.uintp),
        ('re', numba.uintp),
        ('bin_sinc_Δψ', uvw.dtype),
        ('l', lm_type),
        ('m', lm_type),
        ('n_max', lm_type),
        ('ref_freq', ref_freq),
        ('decorrelation', decorr_type)]

    JitBinner = jitclass(spec)(Binner)

    def _impl(time, interval, ant1, ant2, uvw,
            ref_freq, chan_freq, chan_width,
            lm_max=1, decorrelation=0.98):
        # 𝞓 𝝿 𝞇 𝞍 𝝼

        if decorrelation < 0.0 or decorrelation > 1.0:
            raise ValueError("0.0 <= decorrelation <= 1.0 must hold")

        l = m = np.sqrt(0.5 * lm_max)
        n_term = 1.0 - l**2 - m**2
        n_max = np.sqrt(n_term) - 1.0 if n_term >= 0.0 else -1.0

        ubl, _, bl_inv, _ = unique_baselines(ant1, ant2)
        utime, _, time_inv, _ = unique_time(time)

        nrow = time.shape[0]
        ntime = utime.shape[0]
        nbl = ubl.shape[0]

        sentinel = np.finfo(time.dtype).max
        binner = JitBinner(0, 0, l, m, n_max, ref_freq, decorrelation)

        # Create the row lookup
        row_lookup = np.full((nbl, ntime), -1, dtype=np.int32)

        for r in range(nrow):
            t = time_inv[r]
            bl = bl_inv[r]

            if row_lookup[bl, t] != -1:
                raise ValueError("Duplicate (TIME, ANTENNA1, ANTENNA2)")

            row_lookup[bl, t] = r

        for bl in range(nbl):
            # Reset the binner for this baseline
            binner.reset()

            for t in range(ntime):
                # Lookup row, continue if non-existent
                r = row_lookup[bl ,t]

                if r == -1:
                    continue

                # We're starting a new bin,
                if binner.bin_count == 0:
                    binner.start_bin(r)
                # Try add the row to the bin
                # If this fails, finalise the bin and start a new one
                elif binner.add_row(r, time, interval, uvw):
                    binner.finalise_bin(uvw, chan_freq, chan_width)
                    binner.start_bin(r)

            # Finalise any remaining data in the bin
            if binner.bin_count > 1:
                binner.finalise_bin(uvw, chan_freq, chan_width)

    return _impl