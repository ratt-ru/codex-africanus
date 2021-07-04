import numpy as np
from numpy.testing import assert_array_equal
import pytest

from africanus.rime.parangles import _discovered_backends

no_casa = 'casa' not in _discovered_backends
no_astropy = 'astropy' not in _discovered_backends


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


def _observation_endpoints(year, month, day, hour_duration):
    """
    Start and end points of an observation starting on
    ``year-month-day`` and of duration ``hour_duration``
    in Modified Julian Date seconds
    """
    start = _modified_julian_date(year, month, day)
    end = start + hour_duration / 24.

    # Convert to seconds
    start *= 86400.
    end *= 86400.

    return (start, end)


@pytest.mark.flaky(min_passes=1, max_runs=3)
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


@pytest.mark.flaky(min_passes=1, max_runs=3)
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
    from africanus.rime.parangles_casa import casa_parallactic_angles
    from africanus.rime.parangles_astropy import astropy_parallactic_angles
    from astropy import units
    from astropy.coordinates import Angle

    obs, rtol = obs_and_tol
    start, end = _observation_endpoints(*obs)

    time = np.linspace(start, end, 5)
    ant = wsrt_ants[:4, :]
    fc = np.array([0., 1.04719755], dtype=np.float64)

    astro_pa = astropy_parallactic_angles(time, ant, fc)
    casa_pa = casa_parallactic_angles(time, ant, fc, zenith_frame='AZELGEO')

    # Convert to angle degrees
    astro_pa = Angle(astro_pa, unit=units.deg).wrap_at(180*units.deg)
    casa_pa = Angle(casa_pa*units.rad, unit=units.deg).wrap_at(180*units.deg)

    # Difference in degrees, wrapped at 180
    diff = np.abs((astro_pa - casa_pa).wrap_at(180*units.deg))
    assert np.all(np.abs(diff) < Angle(rtol))


@pytest.mark.flaky(min_passes=1, max_runs=3)
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
    assert_array_equal(np_pa, np.asarray(da_pa.compute()))
