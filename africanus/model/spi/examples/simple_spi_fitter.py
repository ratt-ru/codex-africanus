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
    p.add_argument('--emaj', type=float)
    p.add_argument('--emin', type=float)
    p.add_argument('--pa', type=float)
    p.add_argument('--threshold', default=25, type=float)
    p.add_argument('--channelweights', type=float)
    p.add_argument('--ncpu', type=int)
    return p

args = create_parser().parse_args()

GD = vars(args)
for key in GD.keys():
    print(key, " = ", GD[key])

beampars = (args.emaj*ARCSEC2RAD, args.emin*ARCSEC2RAD, args.pa)

print(beampars)

# load images
modelhdu = fits.open(args.fitsmodel)

# get frequencies
deltanu = modelhdu[0].header['CDELT4']
refnu = modelhdu[0].header['CRVAL4']
nfreq = modelhdu[0].header['NAXIS4']
deltal = modelhdu[0].header['CDELT1'] * DEG2RAD
deltam = modelhdu[0].header['CDELT2'] * DEG2RAD

freqinterval = nfreq*deltanu
freqs = (refnu - freqinterval/2) + np.arange(0,nfreq)*deltanu

ModelCube = np.asarray(modelhdu[0].data, dtype=np.float64).squeeze()
if args.fitsresidual:
    residhdu = fits.open(args.fitsresidual)
    ResidCube = np.asarray(residhdu[0].data, dtype=np.float64).squeeze()
    rms = np.std(ResidCube.flatten())
    Threshold = rms * args.threshold
    assert ModelCube.shape == ResidCube.shape
else:
    ResidCube = None
    print("This works better with a residual. Ignoring!")
    Threshold = 0.1 * ModelCube.max()

image_shape = ModelCube.shape

# get the Gaussian kernel
nchan, nx, ny = image_shape
if nx % 2:
    xlower = -nx//2 + 1
else:
    xlower = nx//2
xupper = nx//2 +1
if ny % 2:
    ylower = -ny//2 + 1
else:
    ylower = ny//2
yupper = ny//2 +1

x = deltal * np.arange(xlower, xupper)
y = deltam * np.arange(ylower, yupper)
xx, yy = np.meshgrid(x, y)


GaussKern = Gaussian2D(xx, yy, beampars)

import matplotlib.pyplot as plt
plt.imshow(GaussKern)
plt.show()

GaussKernhat = fftpack.fft2(iFs(GaussKern))

# Convolve model with Gaussian kernel
ConvModel = np.zeros_like(ModelCube)
for ch in range(nchan):
    tmp = fftpack.fft2(iFs(ModelCube[ch]))
    ConvModel[ch] = Fs(fftpack.ifft2(tmp * GaussKernhat)).real

# add in residuals if they exist
if ResidCube is not None:
    ConvModel += ResidCube

# for ch in range(nchan):
#     plt.imshow(ConvModel[ch])
#     plt.show()

# get pixels above threshold
MinImage = np.amin(ConvModel, axis=0)
MaskIndices = np.argwhere(MinImage > Threshold)
FitCube = ConvModel[:, MaskIndices[:, 0], MaskIndices[:, 1]]

if args.channelweights:
    weights = args.channelweights
else:
    weights = np.ones(nchan)

if args.ncpu:
    ncpu = args.ncpu
    from multiprocessing.pool import ThreadPool
    dask.config.set(pool=ThreadPool(ncpu))
else:
    import multiprocessing
    ncpu = multiprocessing.cpu_count()

_, ncomps = FitCube.shape
FitCubeDask = da.from_array(FitCube.T.astype(np.float64), chunks=(ncomps//ncpu, nchan))
weightsDask = da.from_array(weights.astype(np.float64), chunks=(nchan))
freqsDask = da.from_array(freqs, chunks=(nchan))

alpha, varalpha, Iref, varIref = fit_spi_components(FitCubeDask, weightsDask,
                                                    freqsDask, refnu,
                                                    dtype=np.float64).compute()

alphamap = np.zeros([nx, ny])
Irefmap = np.zeros([nx, ny])
alphastdmap = np.zeros([nx, ny])
Irefstdmap = np.zeros([nx, ny])

alphamap[MaskIndices[:, 0], MaskIndices[:, 1]] = alpha
Irefmap[MaskIndices[:, 0], MaskIndices[:, 1]] = Iref
alphastdmap[MaskIndices[:, 0], MaskIndices[:, 1]] = np.sqrt(varalpha)
Irefstdmap[MaskIndices[:, 0], MaskIndices[:, 1]] = np.sqrt(varIref)

plt.imshow(alphamap)
plt.colorbar()
plt.show()
plt.imshow(Irefmap)
plt.colorbar()
plt.show()
