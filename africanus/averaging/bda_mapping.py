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


@njit(nogil=True, cache=True)
def _impl(time, interval, ant1, ant2, uvw, lm_max=1, decorrelation=0.98):
    l = m = np.sqrt(0.5 * lm_max)
    ubl, _, bl_inv, _ = unique_baselines(ant1, ant2)
    utime, _, time_inv, _ = unique_time(time)

    ntime = utime.shape[0]
    nbl = ubl.shape[0]

    sentinel = np.finfo(time.dtype).max

    shape = (nbl, ntime)
    row_lookup = np.full(shape, -1, dtype=np.int32)

    for r in range(uvw.shape[0]):
        t = time_inv[r]
        bl = bl_inv[r]

        if row_lookup[bl, t] != -1:
            raise ValueError("Duplicate (TIME, ANTENNA1, ANTENNA2)")

        row_lookup[bl, t] = r

    for bl in range(nbl):
        tbin = numba.int32(0)
        bin_count = numba.int32(0)
        bin_flag_count = numba.int32(0)
        bin_time_low = time.dtype.type(0)
        bin_u_low = uvw.dtype.type(0)
        bin_v_low = uvw.dtype.type(0)

        for t in range(ntime):
            r = row_lookup[bl ,t]

            if r == -1:
                continue

            half_int = interval[r] * 0.5

            # We're starting a new bin anyway,
            # just set the lower bin value
            if bin_count == 0:
                bin_time_low = time[r] - half_int
                bin_u_low = uvw[r, 0]
                bin_v_low = uvw[r, 1]
            # If we exceed decorrelation in the bin
            # normalise and start a new bin
            else:
                dt = time[r] + half_int - bin_time_low
                du_dt = l * (uvw[r, 0] - bin_u_low) / dt
                dv_dt = m * (uvw[r, 1] - bin_v_low) / dt

                𝞇 = np.pi * (du_dt + dv_dt)
                sinc_𝞇 = 1.0 if 𝞇 == 0.0 else np.sin(𝞇) / 𝞇

                # Contents of the bin are decorrelated anyway,
                # finish it off and start a new one
                if sinc_𝞇 < decorrelation:
                    print(bl, t, tbin, "outside", sinc_𝞇)
                    tbin += 1
                    bin_count = 0
                    bin_time_low = time[r] - half_int
                    bin_u_low = uvw[r, 0]
                    bin_v_low = uvw[r, 1]
                    bin_flag_count = 0
                else:
                    u = (uvw[r, 0] - bin_u_low) / 2
                    v = (uvw[r, 1] - bin_v_low) / 2
                    phase = np.exp(-2 * np.pi * 1j * (u*l + v*m)).real

                    # Newton Rhapson to find sinc_𝞍
                    y = decorrelation / sinc_𝞇
                    eps = 1.0
                    x = prev_x = np.pi

                    while np.abs(eps) > 1e-12:
                        sinc_x = 1.0 if prev_x == 0.0 else np.sin(prev_x) / prev_x
                        dsinc_x = np.cos(prev_x) / x - np.sin(x) / (x ** 2)

                        eps = sinc_x - y
                        x = prev_x - eps / dsinc_x
                        # print(y, prev_x, x, eps)
                        prev_x = x


                    𝞍 = x / np.pi
                    #sinc_𝞍 = 1.0 if 𝞍 == 0.0 else np.sin(𝞍) / 𝞍
                    sinc_𝞍 = sinc_x
                    
                    print(bl, t, tbin, y, "within", decorrelation, sinc_𝞇, sinc_𝞍, sinc_𝞇*sinc_𝞍, phase)


            bin_count += 1
            

def atemkeng_mapper(time, interval, ant1, ant2, uvw):
    _impl(time, interval, ant1, ant2, uvw)