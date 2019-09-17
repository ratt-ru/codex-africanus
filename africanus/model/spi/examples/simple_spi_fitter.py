#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import dask
import dask.array as da
import numpy as np
from scipy import fftpack
from astropy.io import fits
from africanus.constants.consts import DEG2RAD, ARCSEC2RAD
from africanus.model.spi.dask import fit_spi_components
iFs = np.fft.ifftshift
Fs = np.fft.fftshift

def Gaussian2D(xin, yin, GaussPar=(1., 1., 0.)):
    S0, S1, PA = GaussPar
    PA = 90 + PA
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
    R = np.einsum('nb,bc,cn->n', x.T, A, x, optimize=True)
    return np.ascontiguousarray(np.exp(-2.0*R).reshape(sOut),
                                dtype=np.float64)

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--fitsmodel", type=str)
    p.add_argument("--fitsresidual", type=str)
    p.add_argument('--outfile', type=str, 
                   help="Path to output directory." 
                        "Placed next to input model "
                        "if outfile not provided.")
    p.add_argument('--emaj', default=None, type=float,
                   help="Major axis of restoring beam "
                        "ellipse in degrees. By default "
                        "will try to pinch this from the header.")
    p.add_argument('--emin', default=None, type=float,
                   help="Minor axis of restoring beam ellipse "
                        "in degrees. By default will try to "
                        "pinch this from the header.")
    p.add_argument('--pa', default=None, type=float,
                   help="Position angle of restoring beam ellipse "
                        "in degrees. By default will try to pinch "
                        "this from the header.")
    p.add_argument('--threshold', default=5, type=float,
                   help="Multiple of the rms in the residual to threshold "
                        "on. Only components above threshols*rms will be "
                        "fitted")
    p.add_argument('--maxDR', default=100, type=float,
                   help="Maximum dynamic range used to determine the "
                        "threshold above which components need to be fit."
                        "Only used if residual is not passed in.")
    p.add_argument('--channelweights', default=None, type=float,
                   help='The sum of the weights in each imaging band')
    p.add_argument('--ncpu', default=0, type=int,
                   help="Number of threads to use. Default of zero means "
                        "use all threads")
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

ref_hdr = fits.getheader(args.fitsresidual)
if args.emaj is None:
    print("Attempting to take beampars from fits")
    emaj = ref_hdr['BMAJ1']
    emin = ref_hdr['BMIN1']
    pa = ref_hdr['BPA1']
    print("Success! Using emaj = %f, emin = %f, PA = %f "%(emaj, emin, pa))
else:
    emaj = args.emaj
    emin = args.emin
    pa = args.pa


# load images
model = np.ascontiguousarray(fits.getdata(args.fitsmodel).squeeze(),
                             dtype=np.float64)
mhdr = fits.getheader(args.fitsmodel)  # pinch this header

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

print("Image shape = ", (npix_l, npix_m))

# get frequencies
if mhdr["CTYPE4"] == 'FREQ':
    freq_axis = 4
    nband = mhdr['NAXIS4']
    refpix_nu = mhdr['CRPIX4']
    delta_nu = mhdr['CDELT4']  # assumes units are Hz
    ref_freq = mhdr['CRVAL4']
    ncorr = mhdr['NAXIS3']
elif mhdr["CTYPE3"] == 'FREQ':
    freq_axis = 3
    nband = mhdr['NAXIS3']
    refpix_nu = mhdr['CRPIX3']
    delta_nu = mhdr['CDELT3']  # assumes units are Hz
    ref_freq = mhdr['CRVAL3']
    ncorr = mhdr['NAXIS4']
else:
    raise ValueError("Freq axis must be 3rd or 4th")

if ncorr > 1:
    raise ValueError("Only Stokes I cubes supported")

freqs = ref_freq + np.arange(1 - refpix_nu, 1 + nband - refpix_nu) * delta_nu

print("Reference frequency is %f Hz "%ref_freq)

# get the Gaussian kernel
print("Gausskern")
xx, yy = np.meshgrid(l_coord, m_coord)
gausskern = Gaussian2D(xx, yy, (emaj, emin, pa))
del xx, yy, l_coord, m_coord

print("FFT's")
# take FFT
npad_l = int(0.15*npix_l)
npad_m = int(0.15*npix_m)
print("FFT shape = ", (npix_l + 2*npad_l, npix_m + 2*npad_m))
import pyfftw
xhat = pyfftw.empty_aligned([npix_l + 2*npad_l, npix_m + 2*npad_m],
                            dtype='complex128')
FFT = pyfftw.FFTW(xhat, xhat, axes=(0, 1), direction='FFTW_FORWARD',
                  threads=ncpu, flags=('FFTW_ESTIMATE', ))
iFFT = pyfftw.FFTW(xhat, xhat, axes=(0, 1), direction='FFTW_BACKWARD',
                  threads=ncpu, flags=('FFTW_ESTIMATE', ))
xhat[...] = iFs(np.pad(gausskern, ((npad_l, npad_l), (npad_m, npad_l)),
                mode='constant'))
FFT()
gausskernhat = np.copy(xhat)

# Convolve model with Gaussian kernel
for ch in range(nband):
    xhat[...] = iFs(np.pad(model[ch], ((npad_l, npad_l), (npad_m, npad_m)),
                    mode='constant'))
    FFT()
    xhat *= gausskernhat
    iFFT()
    model[ch] = np.copy(Fs(xhat)[npad_l: -npad_l, npad_m: -npad_m].real)
    print("ch = %i, max of convmodel = %f"%(ch, model[ch].max()))

# add in residuals if they exist
if args.fitsresidual is not None:
    resid = fits.getdata(args.fitsresidual).squeeze().astype(np.float64)
    rms = np.std(resid)
    if args.channelweights is None:
        rms_cube = np.std(resid.reshape(nband, npix_l*npix_m), axis=1).ravel()
    threshold = args.threshold * rms
    # model += resid
    del resid
else:
    print("No residual provided. Setting  threshold i.t.o dynamic range")
    threshold = model.max()/args.maxDR
    if args.channelweights is None:
        rms_cube = None

print("Threshold set to %f Jy." % threshold)

# get pixels above threshold
minimage = np.amin(model, axis=0)
maskindices = np.argwhere(minimage > threshold)
del minimage
fitcube = model[:, maskindices[:, 0], maskindices[:, 1]].T

if args.channelweights:
    weights = args.channelweights.astype(np.float64)
else:
    if rms_cube is not None:
        print("Using RMS in each imaging band to determine weights")
        weights = np.where(rms_cube>0, 1.0/rms_cube**2, 0.0)
        # normalise
        weights /= weights.max()
    else:
        print("No weights or residual provided. Using equal weights")
        weights = np.ones(nband, dtype=np.float64)

ncomps, _ = fitcube.shape
fitcube = da.from_array(fitcube.astype(np.float64),
                        chunks=(ncomps//ncpu, nband))
weights = da.from_array(weights.astype(np.float64), chunks=(nband))
freqsdask = da.from_array(freqs.astype(np.float64), chunks=(nband))

print("Fitting %i components"%ncomps)
alpha, _, Iref, _ = fit_spi_components(fitcube, weights, freqsdask,
                                       np.float64(ref_freq)).compute()

alphamap = np.zeros([npix_l, npix_m])
i0map = np.zeros([npix_l, npix_m])

alphamap[maskindices[:, 0], maskindices[:, 1]] = alpha
i0map[maskindices[:, 0], maskindices[:, 1]] = Iref

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

# get the reconstructed cube
Irec_cube = i0map[None, :, :] * (freqs[:, None, None]/ref_freq)**alphamap[None, :, :]
# save it
hdu = fits.PrimaryHDU(header=mhdr)
if freq_axis==3:
    hdu.data = Irec_cube[None, :, :, :]
elif freq_axis==4:
    hdu.data = Irec_cube[:, None, :, :]
hdu.writeto(outfile + 'Irec_cube.fits', overwrite=True)

hdr_keys = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3',
            'NAXIS4', 'BUNIT', 'BMAJ', 'BMIN', 'BPA', 'EQUINOX', 'BTYPE',
            'TELESCOP', 'OBSERVER', 'OBJECT', 'ORIGIN', 'CTYPE1', 'CTYPE2',
            'CTYPE3', 'CTYPE4', 'CRPIX1', 'CRPIX2', 'CRPIX3', 'CRPIX4',
            'CRVAL1', 'CRVAL2', 'CRVAL3', 'CRVAL4', 'CDELT1', 'CDELT2',
            'CDELT3', 'CDELT4', 'CUNIT1', 'CUNIT2', 'CUNIT3', 'CUNIT4',
            'SPECSYS', 'DATE-OBS']

new_hdr={}
for key in hdr_keys:
    new_hdr[key] = ref_hdr[key]

if freq_axis == 3:
    new_hdr["NAXIS3"] = 1
    new_hdr["CRVAL3"] = ref_freq
elif freq_axis == 4:
    new_hdr["NAXIS4"] = 1
    new_hdr["CRVAL4"] = ref_freq

new_hdr = fits.Header(new_hdr)

# save alpha map
hdu = fits.PrimaryHDU(header=new_hdr)
hdu.data = alphamap
hdu.writeto(outfile + 'alpha.fits', overwrite=True)

# save I0 map
hdu = fits.PrimaryHDU(header=new_hdr)
hdu.data = i0map
hdu.writeto(outfile + 'I0.fits', overwrite=True)

# save clean beam for consistency check
hdu = fits.PrimaryHDU(header=new_hdr)
hdu.data = gausskern
hdu.writeto(outfile + 'clean-beam.fits', overwrite=True)

print("All done here")