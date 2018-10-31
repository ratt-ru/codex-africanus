#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""
import dask.array as da
import numpy as np
from africanus.dft.dask import im_to_vis, vis_to_im
from africanus.reduction.psf_redux import PSF_adjoint, PSF_response, F, sigma_approx
# from astropy.io import fits
import matplotlib.pyplot as plt
from africanus.opts.primaldual import primal_dual_solver as pds
from africanus.opts.data_reader import data_reader, plot


data_path = "/home/antonio/Documents/Masters/Helpful_Stuff/WSCMSSSMFTestSuite/SSMF.MS_p0"
freq = da.array([1.06e9])
NCPU = 8
ra_dec = np.array([[3.15126500e-05], [-0.00551471375]])

uvw_dask, lm_dask, lm_pad_dask, frequency_dask, weights_dask, vis_dask, padding = data_reader(data_path, ra_dec)

wsum = da.sum(weights_dask)
pad_pix = int(da.sqrt(lm_pad_dask.shape[0]))

L = lambda i: im_to_vis(i, uvw_dask, lm_dask, frequency_dask).compute()
LT = lambda v: vis_to_im(v, uvw_dask, lm_dask, frequency_dask).compute()/da.sqrt(wsum)
L_pad = lambda i: im_to_vis(i, uvw_dask, lm_pad_dask, frequency_dask).compute()
LT_pad = lambda v: vis_to_im(v, uvw_dask, lm_pad_dask, frequency_dask).compute()/da.sqrt(wsum)

PSF = LT_pad(weights_dask).reshape([pad_pix, pad_pix])
PSF_hat = F(PSF)

Sigma = abs(sigma_approx(PSF))

half_Sigma = 1/np.sqrt(Sigma)

P = lambda image: PSF_response(image, PSF_hat, half_Sigma)/wsum
PT = lambda image: PSF_adjoint(image, PSF_hat, half_Sigma)/wsum

dirty = LT_pad(vis_dask).reshape([pad_pix, pad_pix])

start = np.zeros_like(dirty)
start[pad_pix//2, pad_pix//2] = 10
cleaned = pds(start, dirty, P, PT, solver='rspd').real[padding:-padding, padding:-padding]/da.sqrt(wsum)

plt.figure('ID')
plt.imshow(dirty[padding:-padding, padding:-padding]/da.sqrt(wsum))
plt.colorbar()

plt.figure('IM')
plt.imshow(cleaned)
plt.colorbar()

plt.show()
