#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

import dask.array as da
import numpy as np
from africanus.dft.dask import im_to_vis, vis_to_im
from africanus.reduction.psf_redux import PSF_adjoint, PSF_response, F, diag_probe
from astropy.io import fits
import xarrayms
import matplotlib.pyplot as plt
from africanus.opts.primaldual import primal_dual_solver as pds
# Test Power method
# from sub_opts import pow_method as pm
# import numpy as np
#
# eig = 0
#
# while eig is 0:
#     A = np.random.randn(10, 10)
#     G = A.T.dot(A)
#     eig_vals = np.linalg.eigvals(G)
#     if min(eig_vals) > 0:
#         eig = max(eig_vals)
#
# spec = pm(G.dot, G.conj().T.dot, [10,1])
#
# print(eig-spec)


# def test_pd():
"""
Tests to see if there is any great difference between the model image image and the resolved image
"""

npix = 65
nrow = 35100
nchan = 1

pad_fact = .5
padding = int(npix*pad_fact)
pad_pix = npix + 2*padding


def radec_to_lm(ra0, dec0, ra, dec):
    delta_ra = ra - ra0
    l = (np.cos(dec) * np.sin(delta_ra))
    m = (np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(delta_ra))
    return l, m


# generate frequencies
frequency = np.array([1.06e9])
ref_freq = 1
freq = frequency/ref_freq

data_path = "/home/antonio/Documents/Masters/Helpful_Stuff/WSCMSSSMFTestSuite/SSMF.MS_p0"

# generate lm-coordinates
ra_pos = 3.15126500e-05
dec_pos = -0.00551471375
l_val, m_val = radec_to_lm(0, 0, ra_pos, dec_pos)
x_range = max(abs(l_val), abs(m_val))*1.2
x = np.linspace(-x_range, x_range, npix)
ll, mm = np.meshgrid(x, x)
lm = np.vstack((ll.flatten(), mm.flatten())).T

pad_range = x_range + padding*(x[1] - x[0])
x_pad = np.linspace(-pad_range, pad_range, pad_pix)
ll_pad, mm_pad = np.meshgrid(x_pad, x_pad)
lm_pad = np.vstack((ll_pad.flatten(), mm_pad.flatten())).T

for ds in xarrayms.xds_from_ms(data_path):
    Vdat = ds.DATA.data.compute()
    uvw = ds.UVW.data.compute()[0:nrow, :]
    weights = ds.WEIGHT.data.compute()[0:nrow, 0:nchan]

vis = Vdat[0:nrow, 0:nchan, 0]

wsum = sum(weights)

chunk = nrow//10

uvw_dask = da.from_array(uvw, chunks=(chunk, 3))
lm_dask = da.from_array(lm, chunks=(npix**2, 2))
lm_pad_dask = da.from_array(lm_pad, chunks=(pad_pix**2, 2))
frequency_dask = da.from_array(freq, chunks=nchan)
vis_dask = da.from_array(vis, chunks=(chunk, nchan))
weights_dask = da.from_array(weights, chunks=(chunk, nchan))

L = lambda image: im_to_vis(image, uvw_dask, lm_dask, frequency_dask).compute()
LT = lambda v: vis_to_im(v, uvw_dask, lm_dask, frequency_dask).compute()/np.sqrt(wsum)
LT_pad = lambda v: vis_to_im(v, uvw_dask, lm_pad_dask, frequency_dask).compute()/np.sqrt(wsum)

PSF = LT_pad(weights_dask).reshape([pad_pix, pad_pix])
PSF_hat = F(PSF)

Temp_Sigma = np.ones(pad_pix**2)

temp_P = lambda image: PSF_response(image, PSF_hat, Temp_Sigma)/np.sqrt(wsum/pad_pix)

Sigma = np.abs(diag_probe(temp_P, pad_pix))

half_Sigma = 1/np.sqrt(Sigma)

P = lambda image: PSF_response(image, PSF_hat, Temp_Sigma)
PT = lambda image: PSF_adjoint(image, PSF_hat, Temp_Sigma)/np.sqrt(wsum/pad_pix)

dirty = LT(vis)
dirty = dirty.reshape([npix, npix])
dirty = np.pad(dirty, padding, 'constant')

start = np.zeros_like(dirty)
start[pad_pix//2, pad_pix//2] = 10
cleaned = pds(start, dirty, P, PT, solver='rspd', maxiter=2000).real[padding:-padding, padding:-padding]/np.sqrt(wsum)

plt.figure('ID')
plt.imshow(dirty[padding:-padding, padding:-padding]/np.sqrt(wsum))
plt.colorbar()

plt.figure('IM')
plt.imshow(cleaned)
plt.colorbar()

plt.show()

hdu = fits.PrimaryHDU(dirty)
hdul = fits.HDUList([hdu])
hdul.writeto('/home/antonio/Documents/Masters/Helpful_Stuff/WSCMSSSMFTestSuite/dirty.fits', overwrite=True)
hdul.close()

hdu = fits.PrimaryHDU(cleaned)
hdul = fits.HDUList([hdu])
hdul.writeto('/home/antonio/Documents/Masters/Helpful_Stuff/WSCMSSSMFTestSuite/recovered.fits', overwrite=True)
hdul.close()
