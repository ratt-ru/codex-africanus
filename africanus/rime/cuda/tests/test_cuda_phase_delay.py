# -*- coding: utf-8 -*-


import numpy as np
import pytest

from africanus.rime import phase_delay as np_phase_delay
from africanus.rime.cuda.phase import phase_delay as cp_phase_delay


@pytest.mark.parametrize("dtype, decimal", [
    (np.float32, 5),
    (np.float64, 6)
])
def test_cuda_phase_delay(dtype, decimal):
    cp = pytest.importorskip('cupy')

    lm = 0.01*np.random.random((10, 2)).astype(dtype)
    uvw = np.random.random((100, 3)).astype(dtype)
    freq = np.linspace(.856e9, 2*.856e9, 70, dtype=dtype)

    cp_cplx_phase = cp_phase_delay(cp.asarray(lm),
                                   cp.asarray(uvw),
                                   cp.asarray(freq))
    np_cplx_phase = np_phase_delay(lm, uvw, freq)

    np.testing.assert_array_almost_equal(cp.asnumpy(cp_cplx_phase),
                                         np_cplx_phase, decimal=decimal)
