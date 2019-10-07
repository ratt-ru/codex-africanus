#!/usr/bin/env python
# -*- coding: utf-8 -*-


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
    p.add_argument("ms", help="Name of MS")
    p.add_argument("--fitsmodel", help="Fits file to predict from")
    p.add_argument("--row_chunks", default=30000, type=int,
                   help="How to chunks up row dimension.")
    p.add_argument("--ncpu", default=0, type=int,
                   help="Number of threads to use for predict")
    p.add_argument("--colname", default="MODEL_DATA",
                   help="Name of column to write data to.")
    p.add_argument('--field', default=0, type=int,
                   help="Field ID to predict to.")
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
ms_freqs = spw_ds.CHAN_FREQ.data[0].compute()
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
l_coord = np.sort(np.arange(1 - refpix_l, 1 + npix_l - refpix_l)*delta_l)

if hdr['CUNIT2'] != "DEG" and hdr['CUNIT2'] != "deg":
    raise ValueError("Image units must be in degrees")
npix_m = hdr['NAXIS2']
refpix_m = hdr['CRPIX2']
delta_m = hdr['CDELT2'] * np.pi/180  # assumes untis are deg
m0 = hdr['CRVAL2'] * np.pi/180
m_coord = np.arange(1 - refpix_m, 1 + npix_m - refpix_m)*delta_m

npix_tot = npix_l * npix_m

# get frequencies
if hdr["CTYPE4"] == 'FREQ':
    nband = hdr['NAXIS4']
    refpix_nu = hdr['CRPIX4']
    delta_nu = hdr['CDELT4']  # assumes units are Hz
    ref_freq = hdr['CRVAL4']
    ncorr = hdr['NAXIS3']
    freq_axis = str(4)
elif hdr["CTYPE3"] == 'FREQ':
    nband = hdr['NAXIS3']
    refpix_nu = hdr['CRPIX3']
    delta_nu = hdr['CDELT3']  # assumes units are Hz
    ref_freq = hdr['CRVAL3']
    ncorr = hdr['NAXIS4']
    freq_axis = str(3)
else:
    raise ValueError("Freq axis must be 3rd or 4th")

freqs = ref_freq + np.arange(1 - refpix_nu, 1 + nband - refpix_nu) * delta_nu

print("Reference frequency is ", ref_freq)

# TODO - need to use convert for this
if ncorr > 1:
    raise ValueError("Currently only works on a single correlation")

# if frequencies do not match we need to reprojects fits cube
if np.any(ms_freqs != freqs):
    print("Warning - reprojecting fits cube to MS freqs. "
          "This uses a lot of memory. ")
    from scipy.interpolate import RegularGridInterpolator
    # interpolate fits cube
    fits_interp = RegularGridInterpolator((freqs, l_coord, m_coord),
                                          model.squeeze(),
                                          bounds_error=False,
                                          fill_value=None)
    # reevaluate at ms freqs
    vv, ll, mm = np.meshgrid(ms_freqs, l_coord, m_coord,
                             indexing='ij')
    vlm = np.vstack((vv.flatten(), ll.flatten(), mm.flatten())).T
    model_cube = fits_interp(vlm).reshape(nchan, npix_l, npix_m)
else:
    model_cube = model

# set up coords for DFT
ll, mm = np.meshgrid(l_coord, m_coord)
lm = np.vstack((ll.flatten(), mm.flatten())).T

# get non-zero components of model
model_cube = model_cube.reshape(nchan, npix_tot)
model_max = np.amax(np.abs(model_cube), axis=0)
idx_nz = np.argwhere(model_max > 0.0).squeeze()
model_predict = np.transpose(model_cube[:, None, idx_nz],
                             [2, 0, 1])
ncomps = idx_nz.size
model_predict = da.from_array(model_predict, chunks=(ncomps, nchan, ncorr))
lm = da.from_array(lm[idx_nz, :], chunks=(ncomps, 2))
ms_freqs = spw_ds.CHAN_FREQ.data

xds = xds_from_ms(args.ms, columns=["UVW", args.colname],
                  chunks={"row": args.row_chunks})[0]
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

# Submit all graph computations in parallel
with ProgressBar():
    dask.compute(write)
