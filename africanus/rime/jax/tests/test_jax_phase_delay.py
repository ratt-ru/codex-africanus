# -*- coding: utf-8 -*-


import numpy as np
import pytest

from africanus.rime.phase import phase_delay as np_phase_delay
from africanus.rime.jax.phase import phase_delay


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_jax_phase_delay(dtype):
    jax = pytest.importorskip('jax')

    np.random.seed(0)

    uvw = np.random.random(size=(100, 3)).astype(dtype)
    lm = np.random.random(size=(10, 2)).astype(dtype)*0.001
    frequency = np.linspace(.856e9, .856e9*2, 64).astype(dtype)

    # Compute complex phase
    np_complex_phase = np_phase_delay(lm, uvw, frequency)
    complex_phase = jax.jit(phase_delay)(lm, uvw, frequency)

    np.testing.assert_array_almost_equal(complex_phase, np_complex_phase)
    expected_ctype = np.result_type(dtype, np.complex64)
    assert np_complex_phase.dtype == expected_ctype
    assert complex_phase.dtype == expected_ctype
