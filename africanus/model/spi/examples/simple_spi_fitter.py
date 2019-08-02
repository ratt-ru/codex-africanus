#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import dask
import dask.array as da
import numpy as np
from astropy.io import fits
from scipy import fftpack
from africanus.constants.consts import DEG2RAD, ARCSEC2RAD
from africanus.model.spi.dask import fit_spi_components
iFs = np.fft.ifftshift
Fs = np.fft.fftshift

def Gaussian2D(xin, yin, GaussPar=(1., 1., 0.)):
    S0, S1, PA = GaussPar
    SMaj = np.max([S0, S1])
    SMin = np.min([S0, S1])
    A = np.array([[1. / SMaj ** 2, 0],
                  [0, 1. / SMin ** 2]])

    c, s, t = np.cos, np.sin, PA
    R = np.array([[c(t), -s(t)],
                  [s(t), c(t)]])
    A = np.dot(np.dot(R.T, A), R)
    sOut = xin.shape
    x = np.array([xin.ravel(), yin.ravel()])
    R = [np.dot(np.dot(x[:, iPix].T, A), x[:, iPix]) for iPix in range(x.shape[-1])]
    return np.exp(-np.array(R)).reshape(sOut)

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--fitsmodel", type=str)
    p.add_argument("--fitsresidual", type=str)
    p.add_argument("--fitsrestored", type=str)
    p.add_argument('--outfile', type=str, help="Path to output directory." 
                                               "Placed next to input model "
                                               "if outfile not provided.")
    p.add_argument('--emaj', default=None, type=float, help="Major axis of restoring beam ellipse in degrees. By default will try to pinch this from the header.")
    p.add_argument('--emin', default=None, type=float, help="Minor axis of restoring beam ellipse in degrees. By default will try to pinch this from the header.")
    p.add_argument('--pa', default=None, type=float, help="Position angle of restoring beam ellipse in degrees. By default will try to pinch this from the header.")
    p.add_argument('--threshold', default=25, type=float, help="Multiple of the rms in the residual to threshold on. Only components above threshols*rms will be fitted")
    p.add_argument('--maxDR', default=100, type=float, help="Maximum dynamic range used to determine the threshold above which components need to be fit."
                                                            "Only used if residual is not passed in")
    p.add_argument('--channelweights', default=None, type=float, help='The sum of the weights in each imaging band')
    p.add_argument('--ncpu', default=0, type=int, help='Number of threads to use. Default of zero means use all threads')
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

if args.emaj is None:
    print("Attempting to take beampars from fits")
    restoredhdu = fits.getheader(args.fitsrestored)
    emaj = restoredhdu['BMAJ0']
    emin = restoredhdu['BMIN0']
    pa = restoredhdu['BPA0']
    print(emaj, emin, pa)
else:
    emaj = args.emaj
    emin = args.emin
    pa = args.pa


# load images
model = fits.getdata(args.fitsmodel).squeeze().astype(np.float64)
mhdr = fits.getheader(args.fitsmodel)  # going to pinch this header

if mhdr['CUNIT1'] != "DEG" and mhdr['CUNIT1'] != "deg":
    raise ValueError("Image units must be in degrees")
npix_l = mhdr['NAXIS1']
refpix_l = mhdr['CRPIX1']
delta_l = mhdr['CDELT1'] 
l_coord = np.arange(1 - refpix_l, 1 + npix_l - refpix_l)*delta_l

if mhdr['CUNIT2'] != "DEG" and mhdr['CUNIT2'] != "deg":
    raise ValueError("Image units must be in degrees")
npix_m = mhdr['NAXIS2']
refpix_m = mhdr['CRPIX2']
delta_m = mhdr['CDELT2'] 
m_coord = np.arange(1 - refpix_m, 1 + npix_m - refpix_m)*delta_m

# get frequencies
if mhdr["CTYPE4"] == 'FREQ':
    nband = mhdr['NAXIS4']
    refpix_nu = mhdr['CRPIX4']
    delta_nu = mhdr['CDELT4']  # assumes units are Hz
    ref_freq = mhdr['CRVAL4']
    ncorr = mhdr['NAXIS3']
elif mhdr["CTYPE3"] == 'FREQ':
    nband = mhdr['NAXIS3']
    refpix_nu = mhdr['CRPIX3']
    delta_nu = mhdr['CDELT3']  # assumes units are Hz
    ref_freq = mhdr['CRVAL3']
    ncorr = mhdr['NAXIS4']
else:
    raise ValueError("Freq axis must be 3rd or 4th")

freqs = ref_freq + np.arange(1 - refpix_nu, 1 + nband - refpix_nu) * delta_nu

print("Reference frequency is ", ref_freq)

# get the Gaussian kernel
xx, yy = np.meshgrid(l_coord, m_coord)


gausskern = Gaussian2D(xx, yy, (emaj, emin, pa)).astype(np.float64)

import matplotlib.pyplot as plt
# plt.imshow(GaussKern)
# plt.show()

gausskernhat = fftpack.fft2(iFs(gausskern))

# Convolve model with Gaussian kernel
convmodel = np.zeros_like(model)
for ch in range(nband):
    tmp = fftpack.fft2(iFs(model[ch]))
    convmodel[ch] = Fs(fftpack.ifft2(tmp * gausskernhat)).real

# add in residuals if they exist
if args.fitsresidual is not None:
    resid = fits.getdata(args.fitsresidual).squeeze().astype(np.float64)
    rms = np.std(resid)
    threshold = args.threshold * rms
    convmodel += resid
else:
    print("No residual provided. Setting  threshold i.t.o dynamic range")
    threshold = model.max()/args.maxDR

# get pixels above threshold
minimage = np.amin(convmodel, axis=0)
maskindices = np.argwhere(minimage > threshold)
fitcube = convmodel[:, maskindices[:, 0], maskindices[:, 1]]
fitcube = np.ascontiguousarray(fitcube.T.astype(np.float64))

if args.channelweights:
    weights = args.channelweights.astype(np.float64)
else:
    weights = np.ones(nband, dtype=np.float64)

ncomps, _ = fitcube.shape
fitcubedask = da.from_array(fitcube, chunks=(ncomps//ncpu, nband))
weightsdask = da.from_array(weights, chunks=(nband))
freqsdask = da.from_array(freqs.astype(np.float64), chunks=(nband))

alpha, varalpha, Iref, varIref = fit_spi_components(fitcubedask, weightsdask,
                                                    freqsdask, ref_freq).compute()

alphamap = np.zeros([npix_l, npix_m])
i0map = np.zeros([npix_l, npix_m])
alphastdmap = np.zeros([npix_l, npix_m])
i0stdmap = np.zeros([npix_l, npix_m])

alphamap[maskindices[:, 0], maskindices[:, 1]] = alpha
i0map[maskindices[:, 0], maskindices[:, 1]] = Iref
alphastdmap[maskindices[:, 0], maskindices[:, 1]] = np.sqrt(varalpha)
i0stdmap[maskindices[:, 0], maskindices[:, 1]] = np.sqrt(varIref)

if args.outfile is None:
    # find last /
    tmp = args.fitsmodel[::-1]
    idx = tmp.find('/')
    outfile = args.fitsmodel[0:-idx]
else:
    outfile = args.outfile

# save alpha map
if not outfile.endswith('/'):
    outfile = args.outfile
    
hdu = fits.PrimaryHDU(header=mhdr)
hdu.data = alphamap
hdu.writeto(outfile + 'alpha.fits', overwrite=True)

# save I0 map
hdu = fits.PrimaryHDU(header=mhdr)
hdu.data = i0map
hdu.writeto(outfile + 'I0.fits', overwrite=True)