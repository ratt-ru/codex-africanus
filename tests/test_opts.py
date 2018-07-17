#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

import numpy as np
#from africanus.dft.dask import im_to_vis, vis_to_im
from africanus.dft.kernels import im_to_vis, vis_to_im
from casacore.tables import table
from astropy.io import fits
import dask.array as da
import pytest
import xarrayms
import matplotlib.pyplot as plt
from africanus.opts.primaldual import primal_dual_solver as pds
from africanus.opts.sub_opts import pow_method

# def test_pd():
"""
Tests to see if there is any great difference between the model image image and the resolved image
"""

npix = 256

# generate lm-coordinates
xmin = -0.005
xmax = 0.005
x = np.linspace(xmin, xmax, npix)
ll, mm = np.meshgrid(x, x)
lm = np.vstack((ll.flatten(), mm.flatten())).T

# generate frequencies
freq = np.array([1.48e9])#.linspace(base_freq, base_freq + Npix_f*cdelt_f, Npix_f)
#print(freq.shape)

data_path = "/home/antonio/Downloads/WSCMSSSMFTestSuite/SSMF.MS_p0"

nrow = 500
nchan = 1

# make sky model for calibration
# t = table(data_path, readonly=True)
# Vdat = t.getcol("DATA")[:, 0:1, 0]
# uvw = t.getcol("UVW")
# weight = t.getcol("WEIGHT")
# weight = np.ones_like(weight)

for ds in xarrayms.xds_from_ms(data_path):
    Vdat = ds.DATA.data.compute()
    uvw = ds.UVW.data.compute()[0:nrow,:]
    weights = ds.WEIGHT.data.compute()[0:nrow, 0:nchan]

vis = ((Vdat[0:nrow, 0:nchan, 0] + Vdat[0:nrow, 0:nchan, 0])/2.0).reshape(nrow, nchan)

# set up dask arrays
uvw_dask = da.from_array(uvw, chunks=(1000, 3))
lm_dask = da.from_array(lm, chunks=(npix, 2))
frequency_dask = da.from_array(freq, chunks=nchan)
vis_dask = da.from_array(vis, chunks=(1000, nchan))
weights_dask = da.from_array(weights, chunks=(1000, nchan))

# PSF = vis_to_im(weights_dask, uvw_dask, lm_dask, frequency_dask).compute()
PSF = vis_to_im(weights, uvw, lm, freq)

wsum = PSF.max()


def L(image):
    return im_to_vis(image, uvw, lm, freq)/wsum


def LT(v):
    return vis_to_im(v, uvw, lm, freq)/wsum

# dirty = vis_to_im(vis_dask, uvw_dask, lm_dask, frequency_dask).compute()/wsum
dirty = LT(vis)

test = pow_method(L, LT, [npix, npix])
print(test)
# dirty_dask = da.from_array(dirty, chunks=npix)

cleaned = pds(dirty, vis, L, LT)

plt.figure('PSF')
plt.imshow(PSF.reshape(npix, npix)/wsum)
plt.colorbar()


plt.figure('ID')
plt.imshow(dirty.reshape(npix, npix))
plt.colorbar()
plt.show()

plt.figure('IM')
plt.imshow(cleaned.reshape(npix, npix))
plt.colorbar()
plt.show()

hdu = fits.PrimaryHDU(dirty.reshape(npix, npix))
hdul = fits.HDUList([hdu])
hdul.writeto('/home/antonio/Downloads/WSCMSSSMFTestSuite/dirty.fits', overwrite=True)
hdul.close()

hdu = fits.PrimaryHDU(cleaned.reshape(npix, npix))
hdul = fits.HDUList([hdu])
hdul.writeto('/home/antonio/Downloads/WSCMSSSMFTestSuite/recovered.fits', overwrite=True)
hdul.close()
