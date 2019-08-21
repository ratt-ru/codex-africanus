#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse

import numpy as np
from astropy.io import fits
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from africanus.dft.dask import im_to_vis
from daskms import xds_from_ms, xds_from_table, xds_to_table


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms")
    p.add_argument("--fitsmodel")
    p.add_argument("--row_chunks", default=4000, type=int)
    p.add_argument("--ncpu", default=0, type=int)
    p.add_argument("--colname", default="MODEL_DATA")
    p.add_argument('--field', default=0, type=int)
    return p


args = create_parser().parse_args()

if args.ncpu:
    ncpu = args.ncpu
    from multiprocessing.pool import ThreadPool
    dask.config.set(pool=ThreadPool(ncpu))
else:
    import multiprocessing
    ncpu = multiprocessing.cpu_count()

print("Using %i threads" % ncpu)

# Get MS frequencies
spw_ds = list(xds_from_table("::".join((args.ms, "SPECTRAL_WINDOW")),
                             group_cols="__row__"))[0]

# Get frequencies in the measurement set
# If these do not match those in the fits
# file we need to interpolate
ms_freqs = spw_ds.CHAN_FREQ.data.compute()
nchan = ms_freqs.size

# load in the fits file
model = fits.getdata(args.fitsmodel)
# get header
hdr = fits.getheader(args.fitsmodel)

# TODO - check that PHASE_DIR in MS matches that in fits
# get image coordinates
if hdr['CUNIT1'] != "DEG" and hdr['CUNIT1'] != "deg":
    raise ValueError("Image units must be in degrees")
npix_l = hdr['NAXIS1']
refpix_l = hdr['CRPIX1']
delta_l = hdr['CDELT1'] * np.pi/180  # assumes untis are deg
l0 = hdr['CRVAL1'] * np.pi/180
l_coord = l0 + np.arange(1 - refpix_l, 1 + npix_l - refpix_l)*delta_l

if hdr['CUNIT2'] != "DEG" and hdr['CUNIT2'] != "deg":
    raise ValueError("Image units must be in degrees")
npix_m = hdr['NAXIS2']
refpix_m = hdr['CRPIX2']
delta_m = hdr['CDELT2'] * np.pi/180  # assumes untis are deg
m0 = hdr['CRVAL2'] * np.pi/180
m_coord = m0 + np.arange(1 - refpix_m, 1 + npix_m - refpix_m)*delta_m

npix_tot = npix_l * npix_m

# get frequencies
if hdr["CTYPE4"] == 'FREQ':
    nband = hdr['NAXIS4']
    refpix_nu = hdr['CRPIX4']
    delta_nu = hdr['CDELT4']  # assumes units are Hz
    ref_freq = hdr['CRVAL4']
    ncorr = hdr['NAXIS3']
elif hdr["CTYPE3"] == 'FREQ':
    nband = hdr['NAXIS3']
    refpix_nu = hdr['CRPIX3']
    delta_nu = hdr['CDELT3']  # assumes units are Hz
    ref_freq = hdr['CRVAL3']
    ncorr = hdr['NAXIS4']
else:
    raise ValueError("Freq axis must be 3rd or 4th")

freqs = ref_freq + np.arange(1 - refpix_nu, 1 + nband - refpix_nu) * delta_nu

print("Reference frequency is ", ref_freq)

# TODO - need to use convert for this
if ncorr > 1:
    raise ValueError("Currently only works on a single correlation")

# if frequencies do not match we fit a power law and interpolate/extrapolatye
if not (ms_freqs != freqs).all():
    print("Frequencies of fits cube do not match those of ms. "
          "Interpolating/extrapoling using power law")
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
        alphas = np.full((1, ncorr, npix_l, npix_m), -0.7, dtype=np.float64)
        I0s = model

    model_predict = I0s * (ms_freqs[:, None, None, None]/ref_freq) ** alphas
else:
    model_predict = model

# set up coords for DFT
ll, mm = np.meshgrid(l_coord, m_coord)
lm = np.vstack((ll.flatten(), mm.flatten())).T
lm = da.from_array(lm, chunks=(npix_tot, 2))
model_predict = np.transpose(model_predict.reshape(nchan, ncorr, npix_tot),
                             [2, 0, 1])
model_predict = da.from_array(model_predict, chunks=(npix_tot, nchan, ncorr))
ms_freqs = spw_ds.CHAN_FREQ.data

# do the predict
writes = []
for xds in xds_from_ms(args.ms,
                       columns=["UVW", args.colname],
                       chunks={"row": args.row_chunks}):
    uvw = xds.UVW.data
    vis = im_to_vis(model_predict, uvw, lm, ms_freqs)

    data = getattr(xds, args.colname)
    if data.shape != vis.shape:
        print("Assuming only Stokes I passed in")
        if vis.shape[-1] == 1 and data.shape[-1] == 4:
            tmp_zero = da.zeros(vis.shape, chunks=(args.row_chunks, nchan, 1))
            vis = da.concatenate((vis, tmp_zero, tmp_zero, vis), axis=-1)
        elif vis.shape[-1] == 1 and data.shape[-1] == 2:
            vis = da.concatenate((vis, vis), axis=-1)
        else:
            raise ValueError("Incompatible corr axes")
        vis = vis.rechunk((args.row_chunks, nchan, data.shape[-1]))

    # Assign visibilities to MODEL_DATA array on the dataset
    xds = xds.assign(**{args.colname: (("row", "chan", "corr"), vis)})
    # Create a write to the table
    write = xds_to_table(xds, args.ms, [args.colname])
    # Add to the list of writes
    writes.append(write)

# Submit all graph computations in parallel
with ProgressBar():
    dask.compute(writes)
