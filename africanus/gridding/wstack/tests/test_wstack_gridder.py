from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from africanus.constants import c as lightspeed
from africanus.filters import convolution_filter
from africanus.coordinates import radec_to_lmn
from africanus.gridding.wstack import (w_stacking_layers,
                                       w_stacking_bins,
                                       grid)


def rf(*a, **kw):
    return np.random.random(*a, **kw)


def rc(*a, **kw):
    return rf(*a, **kw) + 1j*rf(*a, **kw)


def test_w_stacking_bins():
    radec = np.deg2rad([[-1., 1.]])
    lmn = radec_to_lmn(radec, np.asarray([0, 0]))

    w_min = 0.1
    w_max = 10.

    nlayers = w_stacking_layers(w_min, w_max, lmn[:, 0], lmn[:, 1])

    w_bins = w_stacking_bins(w_min, w_max, nlayers)  # noqa


def test_w_stacking_gridder():
    radec = np.asarray([[-np.pi / 4, np.pi / 4]])
    lmn = radec_to_lmn(radec, np.asarray([0, 0]))

    conv_filter = convolution_filter(3, 63, "kaiser-bessel")
    nx = ny = npix = 257
    corr = (2, 2)
    nchan = 4
    nvis = npix*npix

    cell_size = 6  # 6 arcseconds

    vis = rc((nvis, nchan) + corr)

    # Channels of MeerKAT L band
    ref_wave = lightspeed/np.linspace(.856e9, .856e9*2, nchan)

    # Random UVW coordinates
    uvw = rf(size=(nvis, 3)).astype(np.float64)
    uvw[:, :2] *= 1024.
    uvw[:, 2] *= 128
    flags = np.zeros_like(vis, dtype=np.uint8)
    weights = np.ones_like(vis, dtype=np.float64)

    w_min = uvw[:, 2].min()
    w_max = uvw[:, 2].max()

    nlayers = w_stacking_layers(w_min, w_max, lmn[:, 0], lmn[:, 1])
    w_bins = w_stacking_bins(w_min, w_max, nlayers)

    grids = grid(vis, uvw, flags, weights, ref_wave,
                 conv_filter, w_bins,
                 cell_size,
                 nx=nx, ny=ny,
                 grids=None)

    grids = grid(vis, uvw, flags, weights, ref_wave,
                 conv_filter, w_bins,
                 cell_size,
                 nx=nx, ny=ny,
                 grids=grids)
