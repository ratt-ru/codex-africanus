# -*- coding: utf-8 -*-

import numpy as np
import numba

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

from africanus.constants import c as lightspeed
from africanus.util.numba import generated_jit
from africanus.averaging.support import unique_time, unique_baselines

class RowMapperError(Exception):
    pass



def row_mapper(time, uvw, ants, phase_dir, ref_freq, chan_freq, chan_width, lm_max):
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
