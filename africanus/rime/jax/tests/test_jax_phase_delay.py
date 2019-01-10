# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp
import pytest

from africanus.rime.phase import phase_delay as np_phase_delay
from africanus.rime.jax.phase import phase_delay


@pytest.mark.xfail
def test_jax_phase_delay():
    jax = pytest.importorskip('jax')
    np = pytest.importorskip('jax.numpy')

    uvw = onp.random.random(size=(100, 3))
    lm = onp.random.random(size=(10, 2))*0.001
    frequency = np.linspace(.856e9, .856e9*2, 64, endpoint=True)

    # Inputs should all be doubles
    assert uvw.dtype == lm.dtype == frequency.dtype == np.float64

    # Compute complex phase
    complex_phase = jax.jit(phase_delay)(lm, uvw, frequency)
    np_complex_phase = np_phase_delay(lm, uvw, frequency)

    onp.testing.assert_array_almost_equal(complex_phase, np_complex_phase,
                                          decimal=5)

    # numpy gets it right
    assert np_complex_phase.dtype == np.complex128

    # jax gets it wrong when jitting. expected failure here.
    assert complex_phase.dtype == np.complex128
