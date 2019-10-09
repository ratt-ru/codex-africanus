#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import logging

import numpy as np
import pyrap.tables as pt

from africanus.gridding.simple import grid, degrid
from africanus.gridding.util import estimate_cell_size
from africanus.constants import c as lightspeed
from africanus.filters import convolution_filter, taper as filter_taper

logging.basicConfig(level=logging.DEBUG)


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms")
    p.add_argument("-rc", "--row-chunks", default=10000, type=int)
    p.add_argument("-np", "--npix", default=1024, type=int)
    p.add_argument("-sc", "--cell-size", type=float)
    return p


args = create_parser().parse_args()

# Convolution Filter
conv_filter = convolution_filter(3, 7, "kaiser-bessel")

taper = filter_taper("kaiser-bessel", args.npix, args.npix, conv_filter)

# Determine UVW Coordinate extents
query = """
SELECT
MIN([SELECT ABS(UVW[0]) FROM {ms}]) AS ABS_UMIN,
MAX([SELECT ABS(UVW[0]) FROM {ms}]) as ABS_UMAX,
MIN([SELECT ABS(UVW[1]) FROM {ms}]) AS ABS_VMIN,
MAX([SELECT ABS(UVW[1]) FROM {ms}]) as ABS_VMAX,
MIN([SELECT UVW[2] FROM {ms}]) AS WMIN,
MAX([SELECT UVW[2] FROM {ms}]) AS WMAX
""".format(ms=args.ms)

with pt.taql(query) as Q:
    umin = Q.getcol("ABS_UMIN").item()
    umax = Q.getcol("ABS_UMAX").item()
    vmin = Q.getcol("ABS_VMIN").item()
    vmax = Q.getcol("ABS_VMAX").item()
    wmin = Q.getcol("WMIN").item()
    wmax = Q.getcol("WMAX").item()


# Obtain reference wavelength from the first spectral window
with pt.table("::".join((args.ms, "SPECTRAL_WINDOW"))) as SPW:
    freq = SPW.getcol("CHAN_FREQ")[0]
    wavelength = lightspeed / freq

if args.cell_size:
    cell_size = args.cell_size
else:
    cell_size = estimate_cell_size(umax, vmax, wavelength, factor=3,
                                   ny=args.npix, nx=args.npix).max()

logging.info("Chose a cell_size of %.3f arcseconds" % cell_size)


dirty = None
psf = None

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
        dirty = grid(data,
                     uvw,
                     flag,
                     natural_weight,
                     wavelength,
                     conv_filter,
                     cell_size,
                     ny=args.npix, nx=args.npix,
                     grid=dirty)

        # For PSF, flag entire visibilitiy if any correlations are flagged
        psf_flag = np.any(flag, axis=2, keepdims=True)

        # Accumulate PSF using unity visibilities
        psf = grid(np.ones_like(psf_flag, dtype=dirty.dtype),
                   uvw,
                   psf_flag,
                   np.ones_like(psf_flag, dtype=natural_weight.dtype),
                   wavelength,
                   conv_filter,
                   cell_size,
                   ny=2*args.npix, nx=2*args.npix,
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

# Normalise the PSF
psf = psf.real
psf = (psf / psf.max())

# Scale the dirty image by the psf
# x4 because the N**2 FFT normalization factor
# on a square image double the size
dirty = dirty.real / (psf.max() * 4.)

# Apply the taper
dirty /= taper

logging.info("Dirty maximum %.6f" % dirty.max())

# Save image if we have astropy
try:
    from astropy.io import fits
except ImportError:
    pass
else:
    hdu = fits.PrimaryHDU(dirty)
    with fits.HDUList([hdu]) as hdul:
        hdul.writeto('simple-dirty.fits', overwrite=True)


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

        logging.info("Degridding rows %d-%d", r, r + nrow)

        # Get MS data
        uvw = T.getcol("UVW", startrow=r, nrow=nrow)

        # Just use natural weights
        natural_weight = np.ones((nrow, nchan, 1), dtype=np.float64)

        # Produce visibilities for this chunk of UVW coordinates
        vis = degrid(dirty, uvw, natural_weight, wavelength,
                     conv_filter, cell_size)
