# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from africanus.constants import c as lightspeed
from africanus.gridding.facet.grid import degrid, Metadata
from africanus.gridding.facet.spheroidal import wplanes

import numpy as np
import pytest


def rf(*args, **kwargs):
    return np.random.random(*args, **kwargs)


def rc(*args, **kwargs):
    return rf(*args, **kwargs) + 1j*rf(*args, **kwargs)


@pytest.mark.parametrize("support", [11])
@pytest.mark.parametrize("spheroidal_support", [111])
@pytest.mark.parametrize("npix", [1025])
@pytest.mark.parametrize("wlayers", [7])
@pytest.mark.parametrize("maxw", [30000])
@pytest.mark.parametrize("cell_size", [1.3])
@pytest.mark.parametrize("oversampling", [11])
@pytest.mark.parametrize("lm_shift", [(1e-8, 1e-8)])
def test_degridder(support, spheroidal_support, npix,
                   wlayers, maxw, cell_size,
                   oversampling, lm_shift):
    nrow = 10
    nchan = 8
    ncorr = 4

    vis = rc((nrow, nchan, ncorr)).astype(np.complex64)
    flags = np.random.randint(0, 2, (nrow, nchan, ncorr))
    uvw = rf((nrow, 3)) - 0.5
    freqs = np.linspace(.856e9, 2*.856e9, nchan)
    ref_wave = freqs[nchan // 2] / lightspeed

    cu, cv, wcf, wcf_conj = wplanes(wlayers, cell_size, support, maxw,
                                    npix, oversampling,
                                    lm_shift, freqs)

    meta = Metadata(lm_shift[0], lm_shift[1], ref_wave, maxw,
                    oversampling, cell_size, cell_size,
                    cu, cv)

    grid = rc((npix, npix, ncorr)).astype(np.complex128)

    vis = degrid(grid, uvw, flags, freqs, wcf, wcf_conj, meta)
    assert vis.shape == (uvw.shape[0], freqs.shape[0], grid.shape[2])
