from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from africanus.rime import phase_delay as np_phase_delay
from africanus.rime.cuda.phase import phase_delay as cp_phase_delay


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cupy_phase_delay(dtype):
    cp = pytest.importorskip('cupy')

    lm = 0.01*np.random.random((10, 2)).astype(dtype)
    uvw = np.random.random((100, 3)).astype(dtype)
    freq = np.linspace(.856e9, 2*.856e9, 70, dtype=dtype)

    cp_cplx_phase = cp_phase_delay(cp.asarray(lm),
                                   cp.asarray(uvw),
                                   cp.asarray(freq))
    np_cplx_phase = np_phase_delay(lm, uvw, freq)

    assert np.allclose(cp.asnumpy(cp_cplx_phase), np_cplx_phase)
