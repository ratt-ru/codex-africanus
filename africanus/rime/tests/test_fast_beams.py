import numpy as np

from africanus.rime.fast_beam_cubes import beam_cube_dde


def rf(*a, **kw):
    return np.random.random(*a, **kw)


def rc(*a, **kw):
    return rf(*a, **kw) + 1j*rf(*a, **kw)


def test_fast_beams():
    beam_lw = 10
    beam_mh = 10
    beam_nud = 10

    src = 10
    time = 5
    ants = 4
    chans = 8

    # Source Transform variables
    lm = np.random.random(size=(src, 2))
    parangles = np.random.random(size=(time, ants))
    point_errors = np.random.random(size=(time, ants, 2))
    antenna_scaling = np.random.random(size=(ants, chans, 2))
    freqs = np.linspace(.856e9 - .1e9, .856e9*2 + .1e9, chans)
    beam_freq_map = np.linspace(.856e9, .856e9*2, beam_nud)

    # Beam sampling variables
    beam = rc((beam_lw, beam_mh, beam_nud, 2, 2))

    beam_lm_extents = np.asarray([[-1.0, 1.0], [-1.0, 1.0]])

    ddes = beam_cube_dde(beam, beam_lm_extents, beam_freq_map,
                         lm, parangles, point_errors, antenna_scaling,
                         freqs)
