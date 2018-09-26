import xarrayms
from africanus.dft.kernels import im_to_vis, vis_to_im
import matplotlib.pyplot as plt
import numpy as np
from africanus.reduction.psf_redux import diag_probe, F, iF


# rad/dec to lm coordinates (straight from fundamentals)
def radec_to_lm(ra0, dec0, ra, dec):
    delta_ra = ra - ra0
    l = (np.cos(dec) * np.sin(delta_ra))
    m = (np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(delta_ra))
    return l, m


# how big our data set is going to be
npix = 33
nrow = 500
nchan = 1


# generate lm-coordinates
ra_pos = 3.15126500e-05
dec_pos = -0.00551471375
l_val, m_val = radec_to_lm(0, 0, ra_pos, dec_pos)
x_range = max(abs(l_val), abs(m_val))*1.5
x = np.linspace(-x_range, x_range, npix)
ll, mm = np.meshgrid(x, x)
lm = np.vstack((ll.flatten(), mm.flatten())).T

# generate frequencies
frequency = np.array([1.06e9])
ref_freq = 1
freq = frequency/ref_freq

# read in data file (not on the git so must be changed!)
data_path = "/home/antonio/Documents/Masters/Helpful_Stuff/WSCMSSSMFTestSuite/SSMF.MS_p0"
for ds in xarrayms.xds_from_ms(data_path):
    Vdat = ds.DATA.data.compute()
    uvw = ds.UVW.data.compute()[0:nrow, :]
    weights = ds.WEIGHT.data.compute()[0:nrow, 0:nchan]

vis = Vdat[0:nrow, 0:nchan, 0]

c = 2.99792458e8

# normalisation factor (equal to max(PSF))
wsum = sum(weights)

# Turn DFT into lambda functions for easy, single input access
L = lambda image: im_to_vis(image, uvw, lm, freq)
LT = lambda visibility: vis_to_im(visibility, uvw, lm, freq)

# Generate the PSF using DFT
PSF = LT(weights).reshape([npix, npix])

# Add in padding to the PSF as well as lm and FFT
padding = [npix//2, npix//2]
PSF_pad = np.pad(PSF, padding, 'constant')
pad_pix = PSF_pad.shape[0]

lm_delta = x[1] - x[0]
x_pad = np.linspace(-pad_pix//2*lm_delta, (pad_pix//2+1)*lm_delta, pad_pix)
ll, mm = np.meshgrid(x_pad, x_pad)
lm_pad = np.vstack((ll.flatten(), mm.flatten())).T

# Generate FFT and DFT matrices
R = np.zeros([nrow, pad_pix**2], dtype='complex128')
for k in range(nrow):
    u, v, w = uvw[k]

    for j in range(pad_pix**2):
        l, m = lm_pad[j]
        n = np.sqrt(1.0 - l ** 2 - m ** 2) - 1.0
        R[k, j] = np.exp(-2j*np.pi*(freq[0]/c)*(u*l + v*m + w*n))

RH = R.conj().T

w = np.diag(weights.flatten())


def plot(im, stage):
    plt.figure(stage)
    plt.imshow(im.real)
    plt.colorbar()


# Mostly here for reference, calculating the NC of the DFT using the operators and FFT matrix
def M(vec):
    T0 = vec.flatten()
    T1 = R.dot(T0)
    T2 = w.dot(T1)
    T3 = RH.dot(T2).real.reshape([pad_pix, pad_pix])
    return T3/pad_pix


def pad_PSF_probe(vec):
    T0 = vec
    PSF_hat = F(PSF_pad)
    T1 = F(T0)
    T2 = (PSF_hat.conj()*T1)
    T3 = iF(T2).real
    return T3


vec = np.zeros(pad_pix**2).reshape([pad_pix, pad_pix])
vec[pad_pix//2, pad_pix//2] = 1

im_frrf = M(vec).real
im_frrf_unpad = im_frrf[pad_pix//2-npix//2:pad_pix//2+npix//2+1, pad_pix//2-npix//2:pad_pix//2+npix//2+1]
plot(im_frrf_unpad, "M operator")

im_pad = pad_PSF_probe(vec)
im_unpad = im_pad[pad_pix//2-npix//2:pad_pix//2+npix//2+1, pad_pix//2-npix//2:pad_pix//2+npix//2+1]
plot(im_unpad, 'Padded PSF transform')

x = np.linspace(np.min(im_unpad), np.max(im_unpad), im_unpad.size)
plt.figure('Difference between PSF convolution and M')
plt.plot(x, x, 'k')
plt.scatter(im_unpad, im_frrf_unpad, marker='x')

plt.show()
