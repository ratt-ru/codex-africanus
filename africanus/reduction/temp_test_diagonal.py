import xarrayms
from africanus.dft.dask import im_to_vis, vis_to_im
import matplotlib.pyplot as plt
import numpy as np
from africanus.reduction.psf_redux import F, iF, diag_probe, PSF_response
import dask.array as da


# rad/dec to lm coordinates (straight from fundamentals)
def radec_to_lm(ra0, dec0, ra, dec):
    delta_ra = ra - ra0
    l = (np.cos(dec) * np.sin(delta_ra))
    m = (np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(delta_ra))
    return l, m


# how big our data set is going to be
npix = 33
nrow = 1000
nchan = 1

pad_factor = .5
padding = int(npix*pad_factor)
pad_pix = npix + 2*padding

# generate lm-coordinates
ra_pos = 3.15126500e-05
dec_pos = -0.00551471375
l_val, m_val = radec_to_lm(0, 0, ra_pos, dec_pos)
x_range = max(abs(l_val), abs(m_val))*(1 + pad_factor)
x_range = max(abs(l_val), abs(m_val))*1.2
x = np.linspace(-x_range, x_range, npix)
ll, mm = np.meshgrid(x, x)
lm = np.vstack((ll.flatten(), mm.flatten())).T

pad_range = x_range + padding*(x[1] - x[0])
x_pad = np.linspace(-pad_range, pad_range, pad_pix)
ll_pad, mm_pad = np.meshgrid(x_pad, x_pad)
lm_pad = np.vstack((ll_pad.flatten(), mm_pad.flatten())).T


# generate frequencies
frequency = np.array([1.06e9])
ref_freq = 1
freq = frequency/ref_freq

# read in data file (not on the git so must be changed!)
data_path = "/home/antonio/Documents/Masters/Helpful_Stuff/WSCMSSSMFTestSuite/SSMF.MS_p0"
for ds in xarrayms.xds_from_ms(data_path):
    vis = ds.DATA.data.compute()[0:nrow, 0:nchan, 0]
    uvw = ds.UVW.data.compute()[0:nrow, :]
    weights = ds.WEIGHT.data.compute()[0:nrow, 0:nchan]


c = 2.99792458e8

# normalisation factor (equal to max(PSF))
wsum = sum(weights)

# Turn DFT into lambda functions for easy, single input access
chunk = nrow//10

uvw_dask = da.from_array(uvw, chunks=(chunk, 3))
lm_dask = da.from_array(lm, chunks=(npix**2, 2))
lm_pad_dask = da.from_array(lm_pad, chunks=(pad_pix**2, 2))
frequency_dask = da.from_array(freq, chunks=nchan)
vis_dask = da.from_array(vis, chunks=(chunk, nchan))
weights_dask = da.from_array(weights, chunks=(chunk, nchan))

L = lambda image: im_to_vis(image, uvw_dask, lm_dask, frequency_dask).compute()
LT = lambda v: vis_to_im(v, uvw_dask, lm_dask, frequency_dask).compute()/np.sqrt(wsum)
L_pad = lambda image: im_to_vis(image, uvw_dask, lm_pad_dask, frequency_dask).compute()
LT_pad = lambda v: vis_to_im(v, uvw_dask, lm_pad_dask, frequency_dask).compute()/np.sqrt(wsum)

# # Generate FFT and DFT matrices
# R = np.zeros([nrow, pad_pix**2], dtype='complex128')
# FT = np.zeros([pad_pix**2, pad_pix**2], dtype='complex128')
# for k in range(nrow):
#     u, v, w = uvw[k]
#
#     for j in range(pad_pix**2):
#         l, m = lm_pad[j]
#         n = np.sqrt(1.0 - l ** 2 - m ** 2) - 1.0
#         R[k, j] = np.exp(-2j*np.pi*(freq[0]/c)*(u*l + v*m + w*n))
#
# delta = lm_pad[1, 0]-lm_pad[0, 0]
# F_norm = pad_pix**2
# Ffreq = np.fft.fftshift(np.fft.fftfreq(pad_pix, d=delta))
# jj, kk = np.meshgrid(Ffreq, Ffreq)
# jk = np.vstack((jj.flatten(), kk.flatten())).T
#
# for u in range(pad_pix**2):
#     l, m = lm_pad[u]
#     for v in range(pad_pix**2):
#         j, k = jk[v]
#         FT[u, v] += np.exp(-2j*np.pi*(j*l + k*m))/np.sqrt(F_norm)
#
# R.tofile('R.dat')
# FT.tofile('F.dat')

R = np.fromfile('R.dat', dtype='complex128').reshape([nrow, pad_pix**2])
FT = np.fromfile('F.dat', dtype='complex128').reshape([pad_pix**2, pad_pix**2])


# Generate adjoint matrices (conjugate transpose)
RH = R.conj().T/np.sqrt(wsum)
FH = FT.conj().T

w = np.diag(weights.flatten())

# Generate the PSF using DFT
PSF = LT_pad(weights_dask).reshape([pad_pix, pad_pix])


def plot(im, stage):
    plt.figure(stage)
    plt.imshow(im.reshape([npix, npix]).real)
    plt.colorbar()


# Mostly here for reference, calculating the NC of the DFT using the operators and FFT matrix
def M(vec):
    T1 = L(da.from_array(vec.reshape([npix**2, 1]), chunks=(npix**2, 1)))
    T2 = w.dot(T1)
    T3 = LT(T2).real.reshape([npix, npix])
    return T3/np.sqrt(wsum)


def M_pad(vec):
    T1 = L_pad(da.from_array(vec.reshape([pad_pix**2, 1]), chunks=(pad_pix**2, 1)))
    T2 = w.dot(T1)
    T3 = LT_pad(T2).real.reshape([pad_pix, pad_pix])
    return T3/np.sqrt(wsum)


PSF_hat = F(PSF)


# def PSF_probe(vec):
#     T0 = np.pad(vec.reshape([npix, npix]), [padding, padding], 'constant')
#     T1 = F(T0)
#     T2 = PSF_hat * T1
#     T3 = iF(T2).real
#     return T3[padding:-padding, padding:-padding].flatten()*np.sqrt(npix)  #pad_factor#*(pad_pix - npix)/pad_pix


sigma = np.ones(pad_pix**2)
P = lambda image: PSF_response(image, PSF_hat, sigma)*np.sqrt(pad_pix**2/wsum)

# # Test that RH and LT produce the same value
# test_ones = np.ones_like(weights)
# test0 = RH.dot(test_ones).real
# test1 = LT(test_ones).real
# test2 = abs(test0 - test1)
# print("Sum of difference between RH and LT: ", sum(test2.flatten()))
#
# # Test self adjointness of R RH
# gamma1 = np.random.randn(pad_pix**2)
# gamma2 = np.random.randn(weights.size)
#
# LHS = gamma2.T.dot(R.dot(gamma1)).real
# RHS = RH.dot(gamma2).T.dot(gamma1).real
#
# print("Self adjointness of R: ", np.abs(LHS - RHS))

# Test that PSF convolution and M give the same answer
# vec = np.random.random(npix**2).reshape([npix, npix])
vec = np.ones([npix, npix])
# vec = np.zeros([npix, npix])
# vec[npix//2, npix//2] = 1
im_psf = P(np.pad(vec, padding, 'constant')).real[padding:-padding, padding:-padding].flatten()
im_frrf = M(vec).real.flatten()

# Set up y=x line
_min = min(min(im_frrf), min(im_psf))
_max = max(max(im_frrf), max(im_psf))
x = np.linspace(_min, _max, im_frrf.size)
plt.figure('Difference between PSF convolution and M')
plt.plot(x, x, 'k')
plt.scatter(im_frrf, im_psf, marker='x')

########################################################################################################################

# plt.figure('Noise Covariance of DFT')
# plt.imshow(im_frrf.reshape([npix, npix]))
# plt.colorbar()

# doing the probing and calculating the PSF diagonal
D_vec = diag_probe(P, pad_pix).real*np.sqrt(pad_pix)*pad_factor
M_mat = FT.dot(RH.dot(w.dot(R.dot(FH))))/np.sqrt(pad_pix**2*wsum)
# M_mat = RH.dot(w.dot(R))/np.sqrt(wsum)

# M_im = M_mat.imag
M_mat = M_mat.real

# reshaping and removing padding from D_vec created by the PSF
D_vec = D_vec.flatten()
M_vec = np.diagonal(M_mat)

# Set up y=x line
_min = min(min(D_vec), min(M_vec))
_max = max(max(D_vec), max(M_vec))
x = np.linspace(_min, _max, D_vec.shape[0])

# plot diagonal comparison (lots of zeros in here when the probe uses PSF_hat)
plt.figure('Values of PSF NC vs FRRF NC')
plt.plot(x, x, 'k')
plt.scatter(M_vec, D_vec, marker='x')

# plot DFT NC
plt.figure('Noise Covariance of M')
plt.imshow(M_mat)
plt.colorbar()

# # plot DFT NC
# plt.figure('Noise Covariance of Imaginary M')
# plt.imshow(M_im)
# plt.colorbar()

# plot PSF NC from probing
plt.figure('Noise Covariance of PSF')
D = np.diag(D_vec)
plt.imshow(D)
plt.colorbar()
#
plt.show()
