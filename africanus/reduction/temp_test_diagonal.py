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
npix = 30
nrow = 100
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

# normalisation factor (equal to max(PSF))
wsum = sum(weights)

# Turn DFT into lambda functions for easy, single input access
L = lambda image: im_to_vis(image, uvw, lm, freq)
LT = lambda visibility: vis_to_im(visibility, uvw, lm, freq)

# Generate the PSF using DFT
PSF = LT(weights).reshape([npix, npix])

# Add in padding to the images if needed and transform the PSF to make life easier
padding = [npix//2, npix//2]
PSF_pad = np.pad(PSF, padding, mode='constant')
PSF_hat = F(PSF_pad)

# Generate FFT and DFT matrices
R = np.zeros([nrow, npix**2], dtype='complex128')
FT = np.zeros([npix**2, npix**2], dtype='complex128')
for k in range(nrow):
    u, v, w = uvw[k]

    for j in range(npix**2):
        l, m = lm[j]
        n = np.sqrt(1.0 - l ** 2 - m ** 2) - 1.0
        R[k, j] = np.exp(-2j*np.pi*freq[0]*(u*l + v*m + w*n))/np.sqrt(wsum)
#
delta = lm[1, 0]-lm[0, 0]
F_norm = npix**2
Ffreq = np.fft.fftshift(np.fft.fftfreq(npix, d=delta))
jj, kk = np.meshgrid(Ffreq, Ffreq)
jk = np.vstack((jj.flatten(), kk.flatten())).T

for u in range(npix**2):
    l, m = lm[u]
    for v in range(npix**2):
        j, k = jk[v]
        FT[u, v] += np.exp(-2j*np.pi*(j*l + k*m))/np.sqrt(F_norm)

# Generate adjoint matrices (conjugate transpose)
RH = R.conj().T
FH = FT.conj().T

w = np.diag(weights.flatten())


# Mostly here for reference, calculating the NC of the DFT using the operators and FFT matrix
def M(vec):
    T0 = FH.dot(vec)
    T1 = R.dot(T0)
    T2 = w.dot(T1)
    T3 = RH.dot(T2)
    T4 = FT.dot(T3)
    return T4.real


# Calculate NC matrix of
T0 = R.dot(FH)
T1 = w.dot(T0)
T2 = RH.dot(T1)
M_mat = FT.dot(T2).real
M_vec = np.diagonal(M_mat)

# Pad the fourier transform and generate PSF_hat
double_pad = [((npix*2)**2 - npix**2)//2, ((npix*2)**2 - npix**2)//2]
F_pad = np.pad(FT, double_pad, mode='constant')
FH_pad = np.pad(FH, double_pad, mode='constant')
PSF_hat = F_pad.dot(PSF_pad.flatten())


def PSF_probe(vec):
    # p = PSF_hat.flatten()*vec.flatten()
    f_vec = F_pad.dot(vec.reshape(PSF_hat.shape)).flatten()
    f_p = (PSF_hat.flatten()*f_vec).reshape(PSF_hat.shape)
    p = FH_pad.dot(f_p)
    return p.flatten()


im_test = np.where(np.random.random(npix**2) < 0.5, -1, 1).reshape([npix, npix])
im_pad = np.pad(im_test, padding, mode='constant')

im_psf = PSF_probe(im_pad).real
im_psf = im_psf.reshape(im_pad.shape)[padding[0]:-padding[0], padding[1]:-padding[1]]
im_frrf = M(im_test.flatten()).reshape([npix, npix])

# plot PSF NC from probing
x = np.linspace(np.min(im_frrf), np.max(im_frrf), im_frrf.size)
plt.figure('product differences')
plt.plot(x, x, 'k')
plt.scatter(im_psf, im_frrf, marker='x')

# # doing the probing and calculating the PSF diagonal
# D_vec = diag_probe(PSF_probe, PSF_hat.size).real
#
# # reshaping and removing padding from D_vec created by the PSF
# D_vec = D_vec.reshape(im_pad.shape)[padding[0]:-padding[0], padding[1]:-padding[1]]
# D_vec = D_vec.flatten()
#
# # Set up y=x line
# _min = min(min(D_vec), min(M_vec))
# _max = max(max(D_vec), max(M_vec))
# x = np.linspace(_min, _max, D_vec.shape[0])
#
# print(M_vec.shape, D_vec.shape)
#
# # plot diagonal comparison (lots of zeros in here when the probe uses PSF_hat)
# plt.figure('Values of PSF NC vs FRRF NC')
# plt.plot(x, x, 'k')
# plt.scatter(M_vec, D_vec, marker='x')
#
# # plot DFT NC
# plt.figure('Noise Covariance of DFT')
# plt.imshow(M_mat)
# plt.colorbar()
#
# # plot PSF NC from probing
# plt.figure('Noise Covariance of PSF')
# D = np.diag(D_vec)
# plt.imshow(D)
# plt.colorbar()

plt.show()
