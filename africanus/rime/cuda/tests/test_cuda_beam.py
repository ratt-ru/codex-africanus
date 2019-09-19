# -*- coding: utf-8 -*-


import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from africanus.rime import beam_cube_dde as np_beam_cube_dde
from africanus.rime.cuda.beam import beam_cube_dde as cp_beam_cude_dde
from africanus.rime.fast_beam_cubes import freq_grid_interp


@pytest.mark.parametrize("corrs", [(2, 2), (4,)])
def test_cuda_beam(corrs):
    src = 2
    time = 10
    ant = 7
    chan = 8

    beam_lw = 10
    beam_mh = 10
    beam_nud = 4

    cp = pytest.importorskip('cupy')

    beam = (np.random.random((beam_lw, beam_mh, beam_nud) + corrs) +
            np.random.random((beam_lw, beam_mh, beam_nud) + corrs)*1j)

    beam_lm_ext = np.array(([[-0.5, 0.5], [-0.5, 0.5]]))

    lm = np.random.random((src, 2)) - 0.5
    freqs = np.linspace(.856e9, 2*.856e9, chan)
    beam_freq_map = np.linspace(freqs[0], freqs[-1], beam_nud)

    parangles = np.random.random((time, ant))
    point_errors = np.zeros((time, ant, chan, 2))
    ant_scales = np.ones((ant, chan, 2))

    np_ddes = np_beam_cube_dde(beam, beam_lm_ext, beam_freq_map,
                               lm, parangles, point_errors, ant_scales,
                               freqs)

    cp_ddes = cp_beam_cude_dde(cp.asarray(beam),
                               cp.asarray(beam_lm_ext),
                               cp.asarray(beam_freq_map),
                               cp.asarray(lm),
                               cp.asarray(parangles),
                               cp.asarray(point_errors),
                               cp.asarray(ant_scales),
                               cp.asarray(freqs))

    assert np_ddes.shape == cp_ddes.shape
