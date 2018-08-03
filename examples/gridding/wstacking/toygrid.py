#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Implements W-Stacking as described in `WSClean <wsclean_>`_.


1. The range of W coordinates is binned into a linear space
   of ``W-layers``.
1. Grid visibilities onto the ``W-layer`` associated with
   their binned W coordinates.
2. The inverse FFT is applied to each layer.
3. Apply a direction dependent phase shift to each layer.
4. Sum the layers together
5. Apply a final scaling factor.

.. _wsclean: https://academic.oup.com/mnras/article/444/1/606/1010067

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging

import numpy as np
import pyrap.tables as pt

from africanus.coordinates import (radec_to_lmn)
from africanus.gridding.wstack import (grid,
                                       degrid,
                                       w_stacking_layers,
                                       w_stacking_bins,
                                       w_stacking_centroids)
from africanus.constants import c as lightspeed
from africanus.filters import convolution_filter


logging.basicConfig(level=logging.DEBUG)


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms")
    p.add_argument("-rc", "--row-chunks", default=10000, type=int)
    p.add_argument("-np", "--npix", default=1024, type=int)
    p.add_argument("-nw", "--n-wlayers", type=int)
    return p


args = create_parser().parse_args()


# Obtain reference wavelength from the first spectral window
with pt.table("::".join((args.ms, "SPECTRAL_WINDOW"))) as SPW:
    freq = SPW.getcol("CHAN_FREQ")[0]
    ref_wave = lightspeed / freq


# Similarity Theorem (https://www.cv.nrao.edu/course/astr534/FTSimilarity.html)
# Scale UV coordinates
CELL_SIZE = 6  # 6 arc seconds
ARCSEC2RAD = np.deg2rad(1.0/(60.*60.))
UV_SCALE = args.npix * CELL_SIZE * ARCSEC2RAD

# Convolution Filter
conv_filter = convolution_filter(3, 63, "sinc")


# Determine Minimum and Maximum W
query = """
SELECT
MIN([SELECT UVW[2] FROM {ms}]) AS WMIN,
MAX([SELECT UVW[2] FROM {ms}]) AS WMAX
""".format(ms=args.ms)

with pt.taql(query) as Q:
    wmin = Q.getcol("WMIN").item()
    wmax = Q.getcol("WMAX").item()

lmn = radec_to_lmn(np.deg2rad([[-1, -1], [1, 1]]), np.zeros((2,)))

if args.n_wlayers is None:
    w_layers = w_stacking_layers(wmin, wmax, lmn[:, 0], lmn[:, 1])
else:
    w_layers = args.n_wlayers

logging.info("%d W layers", w_layers)
w_bins = w_stacking_bins(wmin*UV_SCALE, wmax*UV_SCALE, w_layers)
w_centroids = w_stacking_centroids(w_bins)

cmin, cmax = lmn

grid_l = np.linspace(cmin[0], cmax[0], args.npix)
grid_m = np.linspace(cmin[1], cmax[1], args.npix)
grid_n = np.sqrt(1. - grid_l[:, None]**2 - grid_m[None, :]**2) - 1

psf_l = np.linspace(cmin[0], cmax[0], 2*args.npix)
psf_m = np.linspace(cmin[1], cmax[1], 2*args.npix)
psf_n = np.sqrt(1. - psf_l[:, None]**2 - psf_m[None, :]**2) - 1

dirties = None
psfs = None

with pt.table(args.ms) as T:
    # For each chunk of rows
    for r in list(range(0, T.nrows(), args.row_chunks)):
        # number of rows to read on this iteration
        nrow = min(args.row_chunks, T.nrows() - r)

        logging.info("Gridding rows %d-%d", r, r + nrow)

        # Get MS data
        data = T.getcol("DATA", startrow=r, nrow=nrow)
        uvw = T.getcol("UVW", startrow=r, nrow=nrow)
        flag = T.getcol("FLAG", startrow=r, nrow=nrow)

        _, nchan, ncorr = data.shape

        # Just use natural weights
        natural_weight = np.ones_like(data, dtype=np.float64)

        # Accumulate visibilities into the dirty image
        dirties = grid(data,
                       uvw*UV_SCALE,
                       flag,
                       natural_weight,
                       ref_wave,
                       conv_filter,
                       w_bins,
                       ny=args.npix, nx=args.npix,
                       grids=dirties)

        # For PSF, flag entire visibility if any correlations are flagged
        psf_flag = np.any(flag, axis=2, keepdims=True)

        # Accumulate PSF using unity visibilities
        psfs = grid(np.ones_like(psf_flag, dtype=dirties[0].dtype),
                    uvw*UV_SCALE,
                    psf_flag,
                    np.ones_like(psf_flag, dtype=natural_weight.dtype),
                    ref_wave,
                    conv_filter,
                    w_bins,
                    ny=args.npix*2, nx=args.npix*2,
                    grids=psfs)

dirty_sum = np.zeros(dirties[0].shape[0:2], dtype=dirties[0].real.dtype)
psf_sum = np.zeros(psfs[0].shape[0:2], dtype=psfs[0].dtype)


for w, (dirty, psf, centroid) in enumerate(zip(dirties, psfs, w_centroids)):
    logging.info("FFTing W-Layer %d", w)

    ncorr = dirty.shape[2]

    # FFT each correlation
    fft = np.empty_like(dirty)
    grid_factor = np.exp(2*np.pi*1j*centroid*(grid_n))

    for c in range(ncorr):
        fft[:, :, c] = np.fft.ifftshift(dirty[:, :, c])
        fft[:, :, c] = np.fft.ifft2(fft[:, :, c])
        fft[:, :, c] = np.fft.fftshift(fft[:, :, c])
        fft[:, :, c] *= grid_factor

    # Dirty image composed of the diagonal correlations
    # (XX: I+Q, YY: I - Q) => X+Y = 2I
    if ncorr == 1:
        dirty = fft[:, :, 0].real
    else:
        dirty = (fft[:, :, 0].real + fft[:, :, ncorr-1].real)*0.5

    dirty_sum += dirty

    # FFT the PSF
    psf_fft = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(psf[:, :, 0])))
    psf_fft = psf_fft*np.exp(2*np.pi*1j*centroid*(psf_n))
    psf_sum += psf_fft

grid_final_factor = (1 + grid_n)  # / (wmax - wmin)
psf_final_factor = (1 + psf_n)  # / (wmax - wmin)

dirty_sum *= grid_final_factor
psf_sum *= psf_final_factor

# Normalised Amplitude
#psf = np.abs(psf_sum.real)
psf = psf.real
psf = psf / psf.max()

# Scale the dirty image by the psf
# x4 because the N**2 FFT normalization factor
# on a square image double the size
dirty = dirty_sum.real / (psf.max() * 4.)

logging.info("Dirty maximum %.6f" % dirty.max())

# Display image if we have matplotlib
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass
else:
    plt.figure()
    plt.imshow(dirty, interpolation="nearest", cmap="cubehelix")
    plt.title("DIRTY")
    plt.colorbar()
    plt.show(True)

# Reverse application of final gridding factor
dirty = (dirty / grid_final_factor)[:, :, None]
vis_grids = []

# FFT and apply inverse factor for each W layer
for w, centroid in enumerate(w_centroids):
    ncorr = dirty.shape[2]

    vis_grid = np.empty(dirty.shape, np.complex64)
    grid_factor = np.exp(2*np.pi*1j*centroid*(grid_n))

    for c in range(ncorr):
        vis_grid[:, :, c] = np.fft.fftshift(dirty[:, :, c])
        vis_grid[:, :, c] = np.fft.fft2(vis_grid[:, :, c])
        vis_grid[:, :, c] = np.fft.ifftshift(vis_grid[:, :, c])
        vis_grid[:, :, c] /= grid_factor

    vis_grids.append(vis_grid)

with pt.table(args.ms) as T:
    # For each chunk of rows
    for r in list(range(0, T.nrows(), args.row_chunks)):
        # number of rows to read on this iteration
        nrow = min(args.row_chunks, T.nrows() - r)

        logging.info("Degridding rows %d-%d", r, r + nrow)

        # Get MS data
        uvw = T.getcol("UVW", startrow=r, nrow=nrow)

        # Just use natural weights
        natural_weight = np.ones((nrow, nchan, 1), dtype=np.float64)

        # Produce visibilities for this chunk of UVW coordinates
        vis = degrid(vis_grids, uvw, natural_weight, ref_wave,
                     conv_filter, w_bins)
