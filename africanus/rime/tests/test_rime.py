#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

import numpy as np

import pytest

from africanus.rime.parangles import _discovered_backends


def rf(*a, **kw):
    return np.random.random(*a, **kw)


def rc(*a, **kw):
    return rf(*a, **kw) + 1j*rf(*a, **kw)


def test_phase_delay():
    from africanus.rime import phase_delay

    uvw = np.random.random(size=(100, 3))
    lm = np.random.random(size=(10, 2))
    frequency = np.linspace(.856e9, .856e9*2, 64, endpoint=True)

    from africanus.constants import minus_two_pi_over_c

    # Test complex phase at a particular index in the output
    uvw_i, lm_i, freq_i = 2, 3, 5

    u, v, w = [1, 2, 3]
    l, m = [0.1, 0.2]
    freq = 0.856e9

    # Set up values in the input
    uvw[uvw_i] = [u, v, w]
    lm[lm_i] = [l, m]
    frequency[freq_i] = freq

    # Compute complex phase
    complex_phase = phase_delay(uvw, lm, frequency)

    # Test singular value vs a point in the output
    n = np.sqrt(1.0 - l**2 - m**2) - 1.0
    phase = minus_two_pi_over_c*(u*l + v*m + w*n)*freq
    assert np.all(np.exp(1j*phase) == complex_phase[lm_i, uvw_i, freq_i])


def test_feed_rotation():
    import numpy as np
    from africanus.rime import feed_rotation

    parangles = np.random.random((10, 5))
    pa_sin = np.sin(parangles)
    pa_cos = np.cos(parangles)

    fr = feed_rotation(parangles, feed_type='linear')
    np_expr = np.stack([pa_cos, pa_sin, -pa_sin, pa_cos], axis=2)
    assert np.allclose(fr, np_expr.reshape(10, 5, 2, 2))

    fr = feed_rotation(parangles, feed_type='circular')
    zeros = np.zeros_like(pa_sin)
    np_expr = np.stack([pa_cos - 1j*pa_sin, zeros,
                        zeros, pa_cos + 1j*pa_sin], axis=2)
    assert np.allclose(fr, np_expr.reshape(10, 5, 2, 2))


def _julian_day(year, month, day):
    """
    Given a Anno Dominei date, computes the Julian Date in days.

    Parameters
    ----------
    year : int
    month : int
    day : int

    Returns
    -------
    float
        Julian Date
    """

    # Formula below from
    # http://scienceworld.wolfram.com/astronomy/JulianDate.html
    # Also agrees with https://gist.github.com/jiffyclub/1294443
    return (367*year - int(7*(year + int((month+9)/12))/4)
            - int((3*(int(year + (month - 9)/7)/100)+1)/4)
            + int(275*month/9) + day + 1721028.5)


def _modified_julian_date(year, month, day):
    """
    Given a Anno Dominei date, computes the Modified Julian Date in days.

    Parameters
    ----------
    year : int
    month : int
    day : int

    Returns
    -------
    float
        Modified Julian Date
    """

    return _julian_day(year, month, day) - 2400000.5


def _observation_endpoints(year, month, date, hour_duration):
    """
    Start and end points of an observation starting on
    ``year-month-day`` and of duration ``hour_duration``
    in Modified Julian Date seconds
    """
    start = _modified_julian_date(year, month, date)
    end = start + hour_duration / 24.

    # Convert to seconds
    start *= 86400.
    end *= 86400.

    return (start, end)


@pytest.fixture
def wsrt_ants():
    """ Westerbork antenna positions """
    return np.array([
           [3828763.10544699,   442449.10566454,  5064923.00777],
           [3828746.54957258,   442592.13950824,  5064923.00792],
           [3828729.99081359,   442735.17696417,  5064923.00829],
           [3828713.43109885,   442878.2118934,  5064923.00436],
           [3828696.86994428,   443021.24917264,  5064923.00397],
           [3828680.31391933,   443164.28596862,  5064923.00035],
           [3828663.75159173,   443307.32138056,  5064923.00204],
           [3828647.19342757,   443450.35604638,  5064923.0023],
           [3828630.63486201,   443593.39226634,  5064922.99755],
           [3828614.07606798,   443736.42941621,  5064923.],
           [3828609.94224429,   443772.19450029,  5064922.99868],
           [3828601.66208572,   443843.71178407,  5064922.99963],
           [3828460.92418735,   445059.52053929,  5064922.99071],
           [3828452.64716351,   445131.03744105,  5064922.98793]],
        dtype=np.float64)


no_casa = 'casa' not in _discovered_backends
no_astropy = 'astropy' not in _discovered_backends


@pytest.mark.parametrize('backend', [
    'test',
    pytest.param('casa', marks=pytest.mark.skipif(
                    no_casa,
                    reason='python-casascore not installed')),
    pytest.param('astropy', marks=pytest.mark.skipif(
                    no_astropy,
                    reason="astropy not installed"))])
@pytest.mark.parametrize('observation', [(2018, 1, 1, 4)])
def test_parallactic_angles(observation, wsrt_ants, backend):
    import numpy as np
    from africanus.rime import parallactic_angles

    start, end = _observation_endpoints(*observation)
    time = np.linspace(start, end, 5)
    ant = wsrt_ants[:4, :]
    fc = np.random.random((2,)).astype(np.float64)

    pa = parallactic_angles(time, ant, fc, backend=backend)
    assert pa.shape == (5, 4)


@pytest.mark.skipif(no_casa or no_astropy,
                    reason="Neither python-casacore or astropy installed")
# Parametrize on observation length and error tolerance
@pytest.mark.parametrize('obs_and_tol', [
    ((2018, 1, 1, 4), "10s"),
    ((2018, 2, 20, 8), "10s"),
    ((2018, 11, 2, 4), "10s")])
def test_compare_astropy_and_casa(obs_and_tol, wsrt_ants):
    """
    Compare astropy and python-casacore parallactic angle implementations.
    More work needs to be done here to get things lined up closer,
    but the tolerances above suggest nothing > 10 arcseconds.
    """
    import numpy as np
    from africanus.rime import parallactic_angles
    from astropy import units
    from astropy.coordinates import Angle

    obs, rtol = obs_and_tol
    start, end = _observation_endpoints(*obs)

    time = np.linspace(start, end, 5)
    ant = wsrt_ants[:4, :]
    fc = np.array([0., 1.04719755], dtype=np.float64)

    astro_pa = parallactic_angles(time, ant, fc, backend='astropy')
    casa_pa = parallactic_angles(time, ant, fc, backend='casa')

    # Convert to angle degrees
    astro_pa = Angle(astro_pa, unit=units.deg).wrap_at(180*units.deg)
    casa_pa = Angle(casa_pa*units.rad, unit=units.deg).wrap_at(180*units.deg)

    # Difference in degrees, wrapped at 180
    diff = np.abs((astro_pa - casa_pa).wrap_at(180*units.deg))
    assert np.all(np.abs(diff) < Angle(rtol))


def test_dask_phase_delay():
    da = pytest.importorskip('dask.array')
    from africanus.rime import phase_delay as np_phase_delay
    from africanus.rime.dask import phase_delay as dask_phase_delay

    uvw = np.random.random(size=(100, 3))
    # So that 1 > 1 - l**2 - m**2 >= 0
    lm = np.random.random(size=(10, 2))*0.01
    frequency = np.linspace(.856e9, .856e9*2, 64, endpoint=True)

    dask_uvw = da.from_array(uvw, chunks=(25, 3))
    dask_lm = da.from_array(lm, chunks=(5, 2))
    dask_frequency = da.from_array(frequency, chunks=16)

    dask_phase = dask_phase_delay(dask_uvw, dask_lm, dask_frequency).compute()
    np_phase = np_phase_delay(uvw, lm, frequency)

    # Should agree completely
    assert np.all(np_phase == dask_phase)


@pytest.mark.parametrize('backend', [
    'test',
    pytest.param('casa', marks=pytest.mark.skipif(
                                no_casa,
                                reason='python-casascore not installed')),
    pytest.param('astropy', marks=pytest.mark.skipif(
                                no_astropy,
                                reason="astropy not installed"))])
@pytest.mark.parametrize('observation', [(2018, 1, 1, 4)])
def test_dask_parallactic_angles(observation, wsrt_ants, backend):
    da = pytest.importorskip('dask.array')
    from africanus.rime import parallactic_angles as np_parangle
    from africanus.rime.dask import parallactic_angles as da_parangle

    start, end = _observation_endpoints(*observation)
    np_times = np.linspace(start, end, 5)
    np_ants = wsrt_ants[:4, :]
    np_fc = np.random.random(size=2)

    np_pa = np_parangle(np_times, np_ants, np_fc, backend=backend)
    np_pa = np.asarray(np_pa)

    da_times = da.from_array(np_times, chunks=(2, 3))
    da_ants = da.from_array(np_ants, chunks=((2, 2), 3))
    da_fc = da.from_array(np_fc, chunks=2)

    da_pa = da_parangle(da_times, da_ants, da_fc, backend=backend)

    assert np.all(np_pa == da_pa.compute())


def test_dask_feed_rotation():
    da = pytest.importorskip('dask.array')
    import numpy as np
    from africanus.rime import feed_rotation as np_feed_rotation
    from africanus.rime.dask import feed_rotation

    parangles = np.random.random((10, 5))
    dask_parangles = da.from_array(parangles, chunks=(5, (2, 3)))

    np_fr = np_feed_rotation(parangles, feed_type='linear')
    assert np.all(np_fr == feed_rotation(dask_parangles, feed_type='linear'))

    np_fr = np_feed_rotation(parangles, feed_type='circular')
    assert np.all(np_fr == feed_rotation(dask_parangles, feed_type='circular'))
