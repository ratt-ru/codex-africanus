#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging

import numpy as np
import pyrap.tables as pt

from africanus.coordinates import (radec_to_lmn)
from africanus.gridding.wstack import (grid,
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
MAX([SELECT UVW[0] FROM {ms}]) AS UMAX,
MAX([SELECT UVW[1] FROM {ms}]) AS VMAX,
MIN([SELECT UVW[2] FROM {ms}]) AS WMIN,
MAX([SELECT UVW[2] FROM {ms}]) AS WMAX
""".format(ms=args.ms)

with pt.taql(query) as Q:
    factor = freq.min() / lightspeed
    umax = Q.getcol("UMAX").item() * factor
    vmax = Q.getcol("UMAX").item() * factor
    wmin = Q.getcol("WMIN").item() * factor
    wmax = Q.getcol("WMAX").item() * factor

lmn = radec_to_lmn(np.deg2rad([[-1, -1], [1, 1]]), np.zeros((2,)))

if args.n_wlayers is None:
    w_layers = w_stacking_layers(wmin, wmax, lmn[:, 0], lmn[:, 1])
else:
    w_layers = args.n_wlayers

logging.info("%d W layers", w_layers)
w_bins = w_stacking_bins(wmin, wmax, w_layers)
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

        # Get MS data
        data = T.getcol("DATA", startrow=r, nrow=nrow)
        uvw = T.getcol("UVW", startrow=r, nrow=nrow)
        flag = T.getcol("FLAG", startrow=r, nrow=nrow)

        _, nchan, ncorr = data.shape

        # Just use natural weights
        natural_weight = np.ones_like(data, dtype=np.float64)

        # Accumulate visibilities into the dirties image
        dirties = grid(data,
                       uvw*UV_SCALE,
                       flag,
                       natural_weight,
                       ref_wave,
                       conv_filter,
                       w_bins,
                       ny=args.npix, nx=args.npix,
                       grids=dirties)

        # For PSF, flag entire visibilitiy if any correlations are flagged
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

    # FFT each correlation and then restack
    fft_shifts = [np.fft.ifftshift(dirty[:, :, p]) for p in range(ncorr)]
    ffts = [np.fft.ifft2(shift) for shift in fft_shifts]
    dirty_fft = [np.fft.fftshift(fft) for fft in ffts]

    dirty_fft = [df*np.exp(2*np.pi*1j*centroid*(grid_n)) for df in dirty_fft]

    # Dirty image composed of the diagonal correlations
    # (XX: I+Q, YY: I - Q) => X+Y = 2I
    if ncorr == 1:
        dirty = dirty_fft[0].real
    else:
        dirty = (dirty_fft[0].real + dirty_fft[ncorr-1].real)*0.5

    dirty_sum += dirty

    # FFT the PSF
    psf_fft = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(psf[:, :, 0])))
    psf_fft = psf_fft*np.exp(2*np.pi*1j*centroid*(psf_n))
    psf_sum += psf_fft

psf = np.abs(psf_sum.real)
psf = psf / psf.max()

dirty = dirty_sum / (psf.max() * 4.)
dirty = dirty * (1 - grid_n) / (wmax - wmin)

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
