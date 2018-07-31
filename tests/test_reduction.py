import xarrayms
from africanus.dft.kernels import im_to_vis, vis_to_im
import numpy as np
import matplotlib.pyplot as plt
from africanus.reduction.psf_redux import  PSF_response as R
from africanus.reduction.psf_redux import PSF_adjoint as RH

def radec_to_lm(ra0, dec0, ra, dec):
    delta_ra = ra - ra0
    l = (np.cos(dec) * np.sin(delta_ra))
    m = (np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(delta_ra))

    return l, m

npix = 257

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

nrow = 1000
nchan = 1

for ds in xarrayms.xds_from_ms(data_path):
    Vdat = ds.DATA.data.compute()
    uvw = ds.UVW.data.compute()[0:nrow, :]
    weights = ds.WEIGHT.data.compute()[0:nrow, 0:nchan]

vis = Vdat[0:nrow, 0:nchan, 0]

wsum = sum(weights)

L = lambda image: im_to_vis(image, uvw, lm, freq)
LT = lambda v: vis_to_im(v, uvw, lm, freq)/wsum

M = lambda im: LT(weights.dot(L(im).T))

PSF = LT(weights).real.reshape([npix, npix])

dirty = np.random.randn(npix, npix)  # LT(vis).real.reshape([npix, npix])
start = np.random.randn(npix, npix)
#
# sandwich = M(start.reshape([npix**2, 1])).reshape([npix, npix]).real
# conv_psf = PSF_response(start, PSF, [npix//2, npix//2])[npix//2:-npix//2+1, npix//2:-npix//2+1]
#
# print(sum(sum(abs(conv_psf - sandwich))))

padding = [npix//2, npix//2]
dirty = np.pad(dirty, padding, mode='constant')
start = np.pad(start, padding, mode='constant')

Rx = R(start, PSF, padding).flatten()
Ry = RH(dirty, PSF, padding).flatten()

LHS = dirty.flatten().T.dot(Rx)
RHS = Ry.T.dot(start.flatten())

print(np.abs(LHS - RHS))
