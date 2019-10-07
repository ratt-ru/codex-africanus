# -*- coding: utf-8 -*-


import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from africanus.rime import beam_cube_dde as np_beam_cube_dde
from africanus.rime.cuda.beam import beam_cube_dde as cp_beam_cude_dde

cp = pytest.importorskip('cupy')


@pytest.mark.parametrize("corrs", [(2, 2), (4,), (2,), (1,)])
def test_cuda_beam(corrs):
    rs = np.random.RandomState(42)

    src, time, ant, chan = 20, 29, 14, 64
    beam_lw = beam_mh = beam_nud = 50

    beam = (rs.normal(size=(beam_lw, beam_mh, beam_nud) + corrs) +
            rs.normal(size=(beam_lw, beam_mh, beam_nud) + corrs)*1j)

    beam_lm_ext = np.array(([[-0.5, 0.5], [-0.5, 0.5]]))

    lm = rs.normal(size=(src, 2)) - 0.5

    if chan == 1:
        freqs = np.array([.856e9*3 / 2])
    else:
        freqs = np.linspace(.856e9, 2*.856e9, chan)

    beam_freq_map = np.linspace(.856e9, 2*.856e9, beam_nud)

    parangles = rs.normal(size=(time, ant))
    point_errors = rs.normal(size=(time, ant, chan, 2))
    ant_scales = rs.normal(size=(ant, chan, 2))

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

    assert_array_almost_equal(np_ddes, cp.asnumpy(cp_ddes))
