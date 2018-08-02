import xarrayms
from africanus.dft.kernels import im_to_vis, vis_to_im
import numpy as np
import matplotlib.pyplot as plt
from africanus.reduction.psf_redux import  PSF_response as R
from africanus.reduction.psf_redux import PSF_adjoint as RH
from africanus.reduction.psf_redux import F, iF

def radec_to_lm(ra0, dec0, ra, dec):
    delta_ra = ra - ra0
    l = (np.cos(dec) * np.sin(delta_ra))
    m = (np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(delta_ra))

    return l, m

npix = 31

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

data_path = "/home/antonio/Documents/Masters/Helpful_Stuff/WSCMSSSMFTestSuite/SSMF.MS_p0"

nrow = 100
nchan = 1

for ds in xarrayms.xds_from_ms(data_path):
    Vdat = ds.DATA.data.compute()
    uvw = ds.UVW.data.compute()[0:nrow, :]
    weights = ds.WEIGHT.data.compute()[0:nrow, 0:nchan]

vis = Vdat[0:nrow, 0:nchan, 0]

wsum = sum(weights)
#
# L = lambda image: im_to_vis(image, uvw, lm, freq)
# LT = lambda v: vis_to_im(v, uvw, lm, freq)
#
# PSF = LT(weights).real.reshape([npix, npix])/wsum
#
# dirty = np.random.randn(npix, npix)
# start = np.random.randn(npix, npix)
#
# padding = [npix//2, npix//2]
# dirty = np.pad(dirty, padding, mode='constant')
# start = np.pad(start, padding, mode='constant')
# PSF_pad = np.pad(PSF, padding, mode='constant')
# PSF_hat = F(PSF_pad)
#
# Rx = R(start, PSF_hat).flatten()
# Ry = RH(dirty, PSF_hat).flatten()
#
# LHS = dirty.flatten().T.dot(Rx)
# RHS = Ry.conj().T.dot(start.flatten())
#
# print(np.abs(LHS - RHS))

R = np.zeros([nrow, npix])
F = np.zeros([npix, npix])

for k in range(nrow):
    u, v, w = uvw[k]

    for j in range(npix):
        l, m = lm[j]
        n = 1 # np.sqrt(1.0 - l ** 2 - m ** 2) - 1.0
        R[k, j] = np.exp(-2j*np.pi*freq[0]*(u*l + v*m + w*(n-1)))

F_norm = npix**2
for k in range(npix):
    for l in range(npix):
        F[k, l] = np.exp(-2j*np.pi*(k*l)/F_norm)/np.sqrt(F_norm)

RH = (R/np.sqrt(wsum)).conj().T
FH = F.conj().T

w = np.diag(weights.flatten())

T0 = R.dot(FH)
T1 = w.dot(T0)
T2 = RH.dot(T1)
covariance = F.dot(T2)

plt.figure('Noise Covariance')
plt.imshow(covariance)
plt.colorbar()
plt.show()
