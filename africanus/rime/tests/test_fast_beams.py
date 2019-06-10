import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest


from africanus.rime.fast_beam_cubes import beam_cube_dde, freq_grid_interp


def rf(*a, **kw):
    return np.random.random(*a, **kw)


def rc(*a, **kw):
    return rf(*a, **kw) + 1j*rf(*a, **kw)


@pytest.fixture
def beam_freq_map():
    return np.array([.5, .56,  .7, .91, 1.0])


@pytest.fixture
def freqs():
    """
    Related to the beam_freq_map fixture.
    Explanation of frequency test values:

    1. One value (0.4) below the beam freq range
    2. One value (0.5) on the first beam freq
    3. One value (1.0) on the last beam freq
    4. One value (1.1) above the beam freq range
    """

    return np.array([.4, .5, .6, .7, .8, .9, 1.0, 1.1])


def test_fast_beams(freqs, beam_freq_map):
    beam_lw = 10
    beam_mh = 10
    beam_nud = beam_freq_map.shape[0]

    src = 10
    time = 5
    ants = 4
    chans = freqs.shape[0]

    # Source Transform variables
    lm = np.random.random(size=(src, 2))
    parangles = np.random.random(size=(time, ants))
    point_errors = np.random.random(size=(time, ants, 2))
    antenna_scaling = np.random.random(size=(ants, chans, 2))

    # Beam variables
    beam = rc((beam_lw, beam_mh, beam_nud, 2, 2))
    beam_lm_extents = np.asarray([[-1.0, 1.0], [-1.0, 1.0]])

    grid_pos, freq_scale, fgrid_diff = freq_grid_interp(freqs, beam_freq_map)

    # Frequencies (first -- 0.8 and last -- 1.1)
    # outside the beam result in scaling,
    assert_array_equal(freq_scale, [0.8, 1.,  1.,  1.,  1.,  1.,  1.,  1.1])
    # Frequencies outside the beam are snapped to 0 if below
    # and beam_nud - 2 if above.
    # Frequencies on the edges are similarly snapped
    assert_array_equal(grid_pos, [0, 0, 1, 2, 2, 2, 3, 3])
    # Frequency less grid position frequency.
    # frequency is snapped to the first and last beam freq value
    # if outside (first and last values)
    # Third frequency value is also exactly on a grid point
    exp_diff = [0., 0., 0.04, 0., 0.1, 0.2, 0.09, 0.09]
    assert_array_almost_equal(fgrid_diff, exp_diff)

    ddes = beam_cube_dde(beam, beam_lm_extents, beam_freq_map,  # noqa
                         lm, parangles, point_errors, antenna_scaling,
                         freqs)
