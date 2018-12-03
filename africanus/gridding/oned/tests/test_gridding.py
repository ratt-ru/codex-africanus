import numpy as np
import pytest


def test_oned_gridding():
    from africanus.gridding.oned.gridding import grid
    from africanus.constants import c as lightspeed
    from africanus.filters.kaiser_bessel_filter import kaiser_bessel

    filter_width = 7
    oversample = 3
    beta = 2.34
    cell_size = 8

    vis = np.asarray([[[1.0 + 2.0j]]])
    uvw = np.asarray([[10.0, 11.0, 12.0]])
    ref_wave = np.asarray([.856e9 / lightspeed])
    width = filter_width*oversample

    u = np.arange(width, dtype=np.float64) - width // 2

    conv_filter = kaiser_bessel(u, width, beta)
    oned_grid = np.empty(21, dtype=np.complex128)

    grid(vis, uvw, ref_wave, conv_filter, oversample, cell_size, oned_grid)



