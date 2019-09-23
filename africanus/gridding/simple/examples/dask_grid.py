#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse

import dask.array as da
import numpy as np
import pyrap.tables as pt

from africanus.gridding.simple.dask import grid, degrid
from africanus.gridding.util import estimate_cell_size
from africanus.constants import c as lightspeed
from africanus.filters import convolution_filter
from daskms import xds_from_ms, xds_from_table


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms")
    p.add_argument("-np", "--npix", default=1024, type=int)
    p.add_argument("-nc", "--chunks", default=10000, type=int)
    p.add_argument("-sc", "--cell-size", type=float)
    return p


args = create_parser().parse_args()

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


xds = list(xds_from_ms(args.ms, chunks={"row": args.chunks}))[0]
spw_ds = list(xds_from_table("::".join((args.ms, "SPECTRAL_WINDOW")),
                             group_cols="__row__"))[0]
wavelength = (lightspeed / spw_ds.CHAN_FREQ.data[0]).compute()


if args.cell_size:
    cell_size = args.cell_size
else:
    cell_size = estimate_cell_size(umax, vmax, wavelength, factor=3,
                                   ny=args.npix, nx=args.npix).max()


# Convolution Filter
conv_filter = convolution_filter(3, 63, "kaiser-bessel")

natural_weights = da.ones_like(xds.DATA.data, dtype=np.float64)

dirty = grid(xds.DATA.data,
             xds.UVW.data,
             xds.FLAG.data,
             natural_weights,
             wavelength,
             conv_filter,
             cell_size,
             ny=args.npix, nx=args.npix)


ncorr = dirty.shape[2]

# FFT each polarisation and then restack
fft_shifts = [da.fft.ifftshift(dirty[:, :, p]) for p in range(ncorr)]
ffts = [da.fft.ifft2(shift) for shift in fft_shifts]
dirty_fft = [da.fft.fftshift(fft) for fft in ffts]

# Flag PSF visibility if any correlations are flagged
psf_flags = da.any(xds.FLAG.data, axis=2, keepdims=True)

# Construct PSF from unity visibilities and natural weights
psf = grid(da.ones_like(psf_flags, dtype=xds.DATA.data.dtype),
           xds.UVW.data,
           psf_flags,
           da.ones_like(psf_flags, dtype=natural_weights.dtype),
           wavelength,
           conv_filter,
           cell_size,
           ny=2*args.npix, nx=2*args.npix)

# Should only be one correlation
assert psf.shape[2] == 1, psf.shape

# FFT the PSF
psf_fft = da.fft.fftshift(da.fft.ifft2(da.fft.ifftshift(psf[:, :, 0])))

# Dirty image composed of the diagonal correlations
if ncorr == 1:
    dirty = dirty_fft[0].real
else:
    dirty = (dirty_fft[0].real + dirty_fft[ncorr - 1].real) * 0.5

# Normalised Amplitude
psf = da.absolute(psf_fft.real)
psf = (psf / da.max(psf))

# Scale the dirty image by the psf
# x4 because the N**2 FFT normalization factor
# on a square image double the size
dirty = dirty / (da.max(psf) * 4.)

# Visualise profiling if we have bokeh
try:
    import bokeh  # noqa
except ImportError:
    from dask.diagnostics import ProgressBar

    with ProgressBar():
        dirty = dirty.compute()
else:
    from dask.diagnostics import ProgressBar
    from dask.diagnostics import Profiler

    with ProgressBar(), Profiler() as prof:
        dirty = dirty.compute()

    prof.visualize()


# Save image if we have astropy
try:
    from astropy.io import fits
except ImportError:
    pass
else:
    hdu = fits.PrimaryHDU(dirty)
    with fits.HDUList([hdu]) as hdul:
        hdul.writeto('simple-dask-dirty.fits', overwrite=True)

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


# Introduce one "correlation" into the dirty image
dirty = dirty[:, :, None]

# Create natural weights for the one correlation
weight_shape = (xds.UVW.shape[0], wavelength.shape[0], 1)
weight_chunks = (xds.UVW.chunks[0], wavelength.shape[0], (1,))
degrid_weights = da.ones(weight_shape, dtype=natural_weights.dtype,
                         chunks=weight_chunks)

# Construct the visibility dask array
vis = degrid(dirty, xds.UVW.data, degrid_weights, wavelength,
             conv_filter, cell_size)

# But only degrid the first 1000 visibilities
vis[:1000].compute()
