#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pyfftw
import argparse
import dask
import dask.array as da
import numpy as np
from astropy.io import fits
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
    fwhm_conv = 2*np.sqrt(2*np.log(2))
    return np.ascontiguousarray(np.exp(-fwhm_conv*R).reshape(sOut),
                                dtype=np.float64)


def create_parser():
    p = argparse.ArgumentParser(description='Simple spectral index fitting'
                                            'tool.',
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--fitsmodel", type=str, required=True)
    p.add_argument("--fitsresidual", type=str)
    p.add_argument('--outfile', type=str,
                   help="Path to output directory. \n"
                        "Placed next to input model if outfile not provided.")
    p.add_argument('--beampars', default=None, nargs='+', type=float,
                   help="Beam parameters matching FWHM of restoring beam "
                        "specified as emaj emin pa. \n"
                        "By default these are taken from the fits header "
                        "of the residual image.")
    p.add_argument('--threshold', default=5, type=float,
                   help="Multiple of the rms in the residual to threshold "
                        "on. \n"
                        "Only components above threshold*rms will be fit.")
    p.add_argument('--maxDR', default=100, type=float,
                   help="Maximum dynamic range used to determine the "
                        "threshold above which components need to be fit. \n"
                        "Only used if residual is not passed in.")
    p.add_argument('--ncpu', default=0, type=int,
                   help="Number of threads to use. \n"
                        "Default of zero means use all threads")
    p.add_argument('--beammodel', default=None, type=str,
                   help="Fits beam model to use. \n"
                        "It is assumed that the pattern is path_to_beam/"
                        "name_corr_re/im.fits. \n"
                        "Provide only the path up to name "
                        "e.g. /home/user/beams/meerkat_lband. \n"
                        "Patterns mathing corr are determined "
                        "automatically. \n"
                        "Only real and imaginary beam models currently "
                        "supported.")
    p.add_argument('--output', default='aiIbc', type=str,
                   help="Outputs to write. Letter correspond to: \n"
                   "a - alpha map \n"
                   "i - I0 map \n"
                   "I - reconstructed cube form alpha and I0 \n"
                   "b - interpolated beam \n"
                   "c - restoring beam used for convolution \n"
                   "Default is to write all of them")
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
if args.beampars is None:
    print("Attempting to take beampars from residual fits header")
    emaj = ref_hdr['BMAJ1']
    emin = ref_hdr['BMIN1']
    pa = ref_hdr['BPA1']
else:
    emaj, emin, pa = args.beampars
print("Using emaj = %3.2e, emin = %3.2e, PA = %3.2e" % (emaj, emin, pa))

# load images
model = np.ascontiguousarray(fits.getdata(args.fitsmodel).squeeze(),
                             dtype=np.float64)
mhdr = fits.getheader(args.fitsmodel)

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

print("Cube frequencies:")
with np.printoptions(precision=2):
    print(freqs)
print("Reference frequency is %3.2e Hz " % ref_freq)

# get the Gaussian convolution kernel
xx, yy = np.meshgrid(l_coord, m_coord)
gausskern = Gaussian2D(xx, yy, (emaj, emin, pa))

# Convolve model with Gaussian restroring beam at lowest frequency
npad_l = int(0.15*npix_l)
npad_m = int(0.15*npix_m)
print("Doing FFT's")
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
    print("ch = %i, max of convmodel = %f" % (ch, model[ch].max()))

# set threshold
if args.fitsresidual is not None:
    resid = fits.getdata(args.fitsresidual).squeeze().astype(np.float64)
    rms = np.std(resid)
    rms_cube = np.std(resid.reshape(nband, npix_l*npix_m), axis=1).ravel()
    threshold = args.threshold * rms
    print("Setting cutoff threshold as %i times the rms "
          "of the residual" % args.threshold)
    del resid
else:
    print("No residual provided. Setting  threshold i.t.o dynamic range. "
          "Max dynamic range is %i" % args.maxDR)
    threshold = model.max()/args.maxDR
    if args.channelweights is None:
        rms_cube = None

print("Threshold set to %f Jy." % threshold)

# get pixels above threshold
minimage = np.amin(model, axis=0)
maskindices = np.argwhere(minimage > threshold)
fitcube = model[:, maskindices[:, 0], maskindices[:, 1]].T

# get primary beam at source locations
if args.beammodel is not None:
    print("Interpolating beam")
    l_source = xx[maskindices[:, 0], maskindices[:, 1]]
    m_source = yy[maskindices[:, 0], maskindices[:, 1]]

    lm_source = np.vstack((l_source.ravel(), m_source.ravel())).T

    ntime = 1
    nant = 1
    parangles = np.zeros((ntime, nant,), dtype=np.float64)
    ant_scale = np.ones((nant, nband, 2))
    freq_scale = np.ones((nband,), dtype=np.float64)
    point_errs = np.zeros((ntime, nant, nband, 2), dtype=np.float64)

    if args.beammodel == "eidos":
        raise NotImplementedError("eidos is coming!!!")
    else:
        print("Loading fits beam patterns from %s" % args.beammodel)
        from glob import glob
        paths = glob(args.beammodel + '_**_**.fits')
        Jones = np.empty((2, 2), dtype=object)
        beam_hdr = None
        for path in paths:
            if 'xx' in path or 'XX' in path or 'rr' in path or 'RR' in path:
                if 're' in path:
                    corr1_re = fits.getdata(path)
                    if beam_hdr is None:
                        beam_hdr = fits.getheader(path)
                elif 'im' in path:
                    corr1_im = fits.getdata(path)
                else:
                    raise NotImplementedError("Only re/im patterns supported")
            elif 'yy' in path or 'YY' in path or 'll' in path or 'LL' in path:
                if 're' in path:
                    corr2_re = fits.getdata(path)
                elif 'im' in path:
                    corr2_im = fits.getdata(path)
                else:
                    raise NotImplementedError("Only re/im patterns supported")
        # get Stokes I amplitude
        beam_amp = (corr1_re**2 + corr1_im**2 + corr2_re**2 + corr2_im**2)/2.0
        # get cube in correct shape for interpolation code
        beam_amp = np.ascontiguousarray(np.transpose(beam_amp, (1, 2, 0))
                                        [:, :, :, None, None])
        # get cube info
        if beam_hdr['CUNIT1'] != "DEG" and beam_hdr['CUNIT1'] != "deg":
            raise ValueError("Image units must be in degrees")
        npix_lb = beam_hdr['NAXIS1']
        refpix_lb = beam_hdr['CRPIX1']
        delta_lb = beam_hdr['CDELT1']
        l_min = (1 - refpix_lb)*delta_lb
        l_max = (1 + npix_lb - refpix_lb)*delta_lb

        if beam_hdr['CUNIT2'] != "DEG" and beam_hdr['CUNIT2'] != "deg":
            raise ValueError("Image units must be in degrees")
        npix_mb = beam_hdr['NAXIS2']
        refpix_mb = beam_hdr['CRPIX2']
        delta_mb = beam_hdr['CDELT2']
        m_min = (1 - refpix_mb)*delta_mb
        m_max = (1 + npix_mb - refpix_mb)*delta_mb

        beam_extents = np.array([[l_min, l_max], [m_min, m_max]])

        # get frequencies
        if beam_hdr["CTYPE3"] != 'FREQ':
            raise ValueError(
                "Cubes are assumed to be in format [nchan, nx, ny]")
        nchan_beam = beam_hdr['NAXIS3']
        refpix_nu = beam_hdr['CRPIX3']
        delta_nu = beam_hdr['CDELT3']  # assumes units are Hz
        beam_ref_freq = beam_hdr['CRVAL3']
        beam_freqs = beam_ref_freq + \
            np.arange(1 - refpix_nu, 1 + nchan_beam - refpix_nu) * delta_nu
        if beam_freqs[0] > freqs[0] or beam_freqs[-1] < freqs[-1]:
            print("Warning - the supplied beam does not have sufficient "
                  "bandwidth. Beam frequencies:")
            with np.printoptions(precision=2):
                print(beam_freqs)

        # LB - dask probably not necessary for small problem
        from africanus.rime.fast_beam_cubes import beam_cube_dde
        beam_source = beam_cube_dde(beam_amp, beam_extents, beam_freqs,
                                    lm_source, parangles, point_errs,
                                    ant_scale, freqs).squeeze()
        # correct cube
        fitcube /= beam_source


# set weights for fit
if rms_cube is not None:
    print("Using RMS in each imaging band to determine weights.")
    weights = np.where(rms_cube > 0, 1.0/rms_cube**2, 0.0)
    # normalise
    weights /= weights.max()
else:
    print("No residual provided. Using equal weights.")
    weights = np.ones(nband, dtype=np.float64)

ncomps, _ = fitcube.shape
fitcube = da.from_array(fitcube.astype(np.float64),
                        chunks=(ncomps//ncpu, nband))
weights = da.from_array(weights.astype(np.float64), chunks=(nband))
freqsdask = da.from_array(freqs.astype(np.float64), chunks=(nband))

print("Fitting %i components" % ncomps)
alpha, _, Iref, _ = fit_spi_components(fitcube, weights, freqsdask,
                                       np.float64(ref_freq)).compute()
print("Done. Writing output.")

alphamap = np.zeros([npix_l, npix_m])
i0map = np.zeros([npix_l, npix_m])
alphamap[maskindices[:, 0], maskindices[:, 1]] = alpha
i0map[maskindices[:, 0], maskindices[:, 1]] = Iref

# save next to model if no outfile is provided
if args.outfile is None:
    # find last /
    tmp = args.fitsmodel[::-1]
    idx = tmp.find('/')
    outfile = args.fitsmodel[0:-idx]
else:
    outfile = args.outfile

hdu = fits.PrimaryHDU(header=mhdr)
if 'I' in args.output:
    # get the reconstructed cube
    Irec_cube = i0map[None, :, :] * \
        (freqs[:, None, None]/ref_freq)**alphamap[None, :, :]
    # save it
    if freq_axis == 3:
        hdu.data = Irec_cube[None, :, :, :]
    elif freq_axis == 4:
        hdu.data = Irec_cube[:, None, :, :]
    name = outfile + 'Irec_cube.fits'
    hdu.writeto(name, overwrite=True)
    print("Wrote reconstructed cube to %s" % name)

if args.beammodel is not None and 'b' in args.output:
    beam_map = np.zeros((nband, npix_l, npix_m))
    beam_map[:, maskindices[:, 0], maskindices[:, 1]] = beam_source.T
    if freq_axis == 3:
        hdu.data = beam_map[None, :, :, :]
    elif freq_axis == 4:
        hdu.data = beam_map[:, None, :, :]
    name = outfile + 'interpolated_beam_cube.fits'
    hdu.writeto(name, overwrite=True)
    print("Wrote interpolated beam cube to %s" % name)

hdr_keys = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3',
            'NAXIS4', 'BUNIT', 'BMAJ', 'BMIN', 'BPA', 'EQUINOX', 'BTYPE',
            'TELESCOP', 'OBSERVER', 'OBJECT', 'ORIGIN', 'CTYPE1', 'CTYPE2',
            'CTYPE3', 'CTYPE4', 'CRPIX1', 'CRPIX2', 'CRPIX3', 'CRPIX4',
            'CRVAL1', 'CRVAL2', 'CRVAL3', 'CRVAL4', 'CDELT1', 'CDELT2',
            'CDELT3', 'CDELT4', 'CUNIT1', 'CUNIT2', 'CUNIT3', 'CUNIT4',
            'SPECSYS', 'DATE-OBS']

new_hdr = {}
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
if 'a' in args.output:
    hdu = fits.PrimaryHDU(header=new_hdr)
    hdu.data = alphamap
    name = outfile + 'alpha.fits'
    hdu.writeto(name, overwrite=True)
    print("Wrote alpha map to %s" % name)

# save I0 map
if 'i' in args.output:
    hdu = fits.PrimaryHDU(header=new_hdr)
    hdu.data = i0map
    name = outfile + 'I0.fits'
    hdu.writeto(name, overwrite=True)
    print("Wrote I0 map to %s" % name)

# save clean beam for consistency check
if 'c' in args.output:
    hdu = fits.PrimaryHDU(header=new_hdr)
    hdu.data = gausskern
    name = outfile + 'clean-beam.fits'
    hdu.writeto(name, overwrite=True)
    print("Wrote clean beam to %s" % name)

print("All done here")
