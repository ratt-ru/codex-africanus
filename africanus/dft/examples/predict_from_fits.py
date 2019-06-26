#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse

import numpy as np
from astropy.io import fits
from astropy import wcs
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from africanus.dft.dask import im_to_vis
from africanus.model.coherency.dask import convert
import xarray as xr
from xarrayms import xds_from_ms, xds_from_table, xds_to_table


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", type=str)
    p.add_argument("--fitsmodel", type=str)
    p.add_argument("--row_chunks", default=10000, type=int)
    p.add_argument("--ncpu", default=0, type=int)
    p.add_argument("--colname", default="MODEL_DATA", type=str)
    return p

args = create_parser().parse_args()

if args.ncpu:
    ncpu = args.ncpu
    import dask
    from multiprocessing.pool import ThreadPool
    dask.config.set(pool=ThreadPool(ncpu))
else:
    import multiprocessing
    ncpu = multiprocessing.cpu_count()

print("Using %i threads"%ncpu)

# Get MS frequencies
spw_ds = list(xds_from_table("::".join((args.ms, "SPECTRAL_WINDOW")),
                             group_cols="__row__"))[0]

# Get frequencies in the measurement set
# If these do not match those in the fits
# file we need to interpolate
ms_freqs = spw_ds.CHAN_FREQ.data
nchan = ms_freqs.compute().size

# load in the fits file
model = fits.getdata(args.fitsmodel)
# get header
hdr = fits.getheader(args.fitsmodel)

# get image coordinates
npix_l = hdr['NAXIS1']
refpix_l = hdr['CRPIX1']
delta_l = hdr['CDELT1'] * np.pi/180  # assumes untis are deg
l_coord = np.arange(1 - refpix_l, 1 + npix_l - refpix_l)*delta_l

npix_m = hdr['NAXIS2']
refpix_m = hdr['CRPIX2']
delta_m = hdr['CDELT2'] * np.pi/180  # assumes untis are deg
m_coord = np.arange(1 - refpix_m, 1 + npix_m - refpix_m)*delta_m

npix_tot = npix_l * npix_m

# get frequencies
nband = hdr['NAXIS4']
refpix_nu = hdr['CRPIX3']
delta_nu = hdr['CDELT4']  # assumes units are Hz
ref_freq = np.float(hdr['CRVAL4'])
freqs = ref_freq + np.arange(1 - refpix_nu, 1 + nband - refpix_nu) * delta_nu

print("Reference frequency is ", ref_freq)

# TODO - need to use convert for this
ncorr = hdr['NAXIS3']
if ncorr > 1:
    print("You will have to convert Stokes to corr for DFT to work properly")

# if frequencies do not match we fit a power law and interpolate/extrapolatye
if ms_freqs.compute() != freqs:
    print("Frequencies of fits cube do not match those of ms. "
          "Interpolating/extrapoling using power law")
    # print("Fits frequencies are ", freqs)
    # print("Interpolating to ", ms_freqs.compute())
    model_predict = np.zeros((nchan, ncorr, npix_l, npix_m), dtype=np.float64)
    alphas = np.zeros((1, ncorr, npix_l, npix_m), dtype=np.float64)
    I0s = np.zeros((1, ncorr, npix_l, npix_m), dtype=np.float64)
    if nband > 1:
        from africanus.model.spi.dask import fit_spi_components
        weights_dask = da.ones(nband, chunks=nband)
        freqs_dask = da.from_array(freqs, chunks=nband)
        for corr in range(ncorr):
            # only fit non-zero components
            model_max = np.amax(np.abs(model[:, corr, :, :]), axis=0)
            Idx = np.argwhere(model_max > 0)
            Ix = Idx[:, 0]
            Iy = Idx[:, 1]
            fitcube = model[:, corr, Ix, Iy].T.astype(np.float64)
            ncomps, _ = fitcube.shape
            model_cube = da.from_array(fitcube, chunks=(ncomps//ncpu, nband))
            
            alpha, _, I0, _ = fit_spi_components(model_cube, weights_dask,
                                                 freqs_dask, ref_freq)
            
            alphas[0, corr, Ix, Iy] = alpha
            I0s[0, corr, Ix, Iy] = I0
    else:
        print("Single frequency fits file. "
              "Assuming spectral index of -0.7 to extrapolate")
        alphas = np.ones((1, ncorr, npix_l, npix_m), dtype=np.float64) * -0.7
        I0s = model
    
    model_predict = I0s * (ms_freqs.reshape(nchan,
                           1, 1, 1)/ref_freq) ** alphas
else:
    model_predict = model

# set up coords for DFT
ll, mm = np.meshgrid(l_coord, m_coord)
lm = np.vstack((ll.flatten(), mm.flatten())).T
lm = da.from_array(lm, chunks=(npix_tot, 2))
# ms_freqs = da.from_array(ms_freqs, chunks=nchan)
model_predict = np.transpose(model_predict.reshape(nchan, ncorr, npix_tot),
                             [2, 0, 1])
model_predict = da.from_array(model_predict, chunks=(npix_tot, nchan, ncorr))

# do the predict
writes = []
for xds in xds_from_ms(args.ms,
                           columns=["UVW", args.colname],
                           chunks={"row": args.row_chunks}):
    uvw = xds.UVW
    vis = im_to_vis(model_predict, uvw, lm, ms_freqs)

    data = getattr(xds, args.colname)
    if data.shape != vis.shape:
        print("Assuming only Stokes I passed in")
        tmp_zero = da.zeros(vis.shape, chunks=(args.row_chunks, nchan, 1))
        vis = da.concatenate((vis, tmp_zero, tmp_zero, vis), axis=-1)
        vis = vis.rechunk((args.row_chunks, nchan, data.shape[-1]))       

    # Assign visibilities to MODEL_DATA array on the dataset
    model_data = xr.DataArray(vis, dims=["row", "chan", "corr"])
    xds = xds.assign(MODEL_DATA=model_data)
    # Create a write to the table
    write = xds_to_table(xds, args.ms, [args.colname])
    # Add to the list of writes
    writes.append(write)

# Submit all graph computations in parallel
with ProgressBar():
    dask.compute(writes)