#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging

import numpy as np
import pyrap.tables as pt

from africanus.gridding.simple import grid, degrid
from africanus.constants import c as lightspeed
from africanus.filters import convolution_filter

logging.basicConfig(level=logging.DEBUG)


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms")
    p.add_argument("-rc", "--row-chunks", default=10000, type=int)
    p.add_argument("-np", "--npix", default=1024, type=int)
    return p


args = create_parser().parse_args()

# Similarity Theorem (https://www.cv.nrao.edu/course/astr534/FTSimilarity.html)
# Scale UV coordinates
CELL_SIZE = 6  # 6 arc seconds
ARCSEC2RAD = np.deg2rad(1.0/(60*60))
UV_SCALE = args.npix * CELL_SIZE * ARCSEC2RAD

# Convolution Filter
conv_filter = convolution_filter(3, 63, "sinc")

# Obtain reference wavelength from the first spectral window
with pt.table("::".join((args.ms, "SPECTRAL_WINDOW"))) as SPW:
    freq = SPW.getcol("CHAN_FREQ")[0]
    ref_wave = lightspeed / freq

dirty = None
psf = None

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

        # Accumulate visibilities into the dirty image
        dirty = grid(data,
                     uvw*UV_SCALE,
                     flag,
                     natural_weight,
                     ref_wave,
                     conv_filter,
                     ny=args.npix, nx=args.npix,
                     grid=dirty)

        # For PSF, flag entire visibilitiy if any correlations are flagged
        psf_flag = np.any(flag, axis=2, keepdims=True)

        # Accumulate PSF using unity visibilities
        psf = grid(np.ones_like(psf_flag, dtype=dirty.dtype),
                   uvw*UV_SCALE,
                   psf_flag,
                   np.ones_like(psf_flag, dtype=natural_weight.dtype),
                   ref_wave,
                   conv_filter,
                   ny=args.npix*2, nx=args.npix*2,
                   grid=psf)

ncorr = dirty.shape[2]

# FFT each correlation and then restack
fft_shifts = [np.fft.ifftshift(dirty[:, :, p]) for p in range(ncorr)]
ffts = [np.fft.ifft2(shift) for shift in fft_shifts]
dirty_fft = [np.fft.fftshift(fft) for fft in ffts]

# Dirty image composed of the diagonal correlations
# (XX: I+Q, YY: I - Q) => X+Y = 2I
if ncorr == 1:
    dirty = dirty_fft[0].real
else:
    dirty = (dirty_fft[0].real + dirty_fft[ncorr-1].real)*0.5

# FFT the PSF
psf_fft = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(psf[:, :, 0])))

# Normalised Amplitude
#psf = np.abs(psf_fft.real)
psf = psf.real
psf = (psf / psf.max())

# Scale the dirty image by the psf
# x4 because the N**2 FFT normalization factor
# on a square image double the size
dirty = dirty.real / (psf.max() * 4.)

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

# Introduce a "correlation"
dirty = dirty[:, :, None]

with pt.table(args.ms) as T:
    # For each chunk of rows
    for r in list(range(0, T.nrows(), args.row_chunks)):
        # number of rows to read on this iteration
        nrow = min(args.row_chunks, T.nrows() - r)

        # Get MS data
        uvw = T.getcol("UVW", startrow=r, nrow=nrow)

        # Just use natural weights
        natural_weight = np.ones((nrow, nchan, 1), dtype=np.float64)

        # Produce visibilities for this chunk of UVW coordinates
        vis = degrid(dirty, uvw, natural_weight, ref_wave, conv_filter)
