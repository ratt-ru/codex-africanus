#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

import numpy as np
from africanus.dft.dask import im_to_vis, vis_to_im
from casacore.tables import table
from astropy.io import fits
import dask.array as da
import pytest

# def test_pd():
"""
Tests to see if there is any great difference between the model image image and the resolved image
"""
with fits.open("/home/antonio/Downloads/WSCMSSSMFTestSuite/output.fits") as the_file:

    model = the_file[0]
    # generate lm-coordinates
    Npix_h = model.data.shape[2]
    Npix_v = model.data.shape[3]
    cdelt_r = model.header['CDELT1']
    cdelt_d = model.header['CDELT2']
    ra = np.linspace((Npix_h/2)*cdelt_r, -(Npix_h/2)*cdelt_r, Npix_h)
    ra_delta = ra - ra[0]
    dec = np.linspace(-(Npix_v/2)*cdelt_r, (Npix_v/2)*cdelt_r, Npix_v)
    l = np.cos(dec) * np.sin(ra_delta)
    m = (np.sin(dec) * np.cos(dec[0]) - np.cos(dec) * np.sin(dec[0]) * np.cos(ra_delta))
    ll, mm = np.meshgrid(l, m)
    lm = np.vstack((ll.flatten(), mm.flatten())).T
    print(lm.shape)

    # generate frequencies
    base_freq = model.header['CRVAL4']
    cdelt_f = model.header['CDELT4']
    Npix_f = 8
    freq = np.array([base_freq])#.linspace(base_freq, base_freq + Npix_f*cdelt_f, Npix_f)
    #print(freq.shape)

    # make sky model for calibration
    t = table("/home/antonio/Downloads/WSCMSSSMFTestSuite/SSMF.MS_p0", readonly=True)
    vis = t.getcol("DATA")[:, 0:1, 0]
    print(vis.shape)
    uvw = t.getcol("UVW")
    print(uvw.shape)
    weight = t.getcol("WEIGHT")
    weight = np.ones_like(weight)

    # set up dask arrays
    uvw_dask = da.from_array(uvw, chunks=(25, 3))
    lm_dask = da.from_array(lm, chunks=(25, 2))
    frequency_dask = da.from_array(freq, chunks=1)
    vis_dask = da.from_array(vis, chunks=(25, 1))

    dirty = vis_to_im(vis_dask, uvw_dask, lm_dask, frequency_dask, np.complex64).compute()

    print(dirty)
