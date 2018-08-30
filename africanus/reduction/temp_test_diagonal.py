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

# Add in padding to the images if needed and transform the PSF to make life easier
padding = [npix//2, npix//2]

# Generate FFT and DFT matrices
R = np.zeros([nrow, npix**2], dtype='complex128')
FT = np.zeros([npix**2, npix**2], dtype='complex128')
for k in range(nrow):
    u, v, w = uvw[k]

    for j in range(npix**2):
        l, m = lm[j]
        n = np.sqrt(1.0 - l ** 2 - m ** 2) - 1.0
        R[k, j] = np.exp(-2j*np.pi*(freq[0]/c)*(u*l + v*m + w*n))
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


def plot(im, stage):
    plt.figure(stage)
    plt.imshow(im.reshape([npix, npix]).real)
    plt.colorbar()


# Mostly here for reference, calculating the NC of the DFT using the operators and FFT matrix
def M(vec):
    T0 = FH.dot(vec)
    T1 = R.dot(T0)
    T2 = w.dot(T1)
    T3 = RH.dot(T2).real
    T4 = FT.dot(T3)
    return T4/npix


def PSF_probe(vec):
    T0 = FH.dot(vec)
    PSF_hat = FT.dot(PSF.flatten())
    T1 = FT.dot(T0)
    T2 = (PSF_hat.conj()*T1)
    T3 = FH.dot(T2).real
    T4 = FT.dot(T3)
    return T4


# Test that RH and LT produce the same value
test_ones = np.ones_like(weights)
test0 = RH.dot(test_ones).real
test1 = LT(test_ones).real
test2 = abs(test0 - test1)
print("Sum of difference between RH and LT: ", sum(test2.flatten()))

# Test self adjointness of R RH
gamma1 = np.random.randn(npix**2)
gamma2 = np.random.randn(weights.size)

LHS = gamma2.T.dot(R.dot(gamma1)).real
RHS = RH.dot(gamma2).T.dot(gamma1).real

print("Self adjointness of R: ", np.abs(LHS - RHS))

# Test self adjointness of FT FH
gamma1 = np.random.randn(npix**2)
gamma2 = np.random.randn(npix**2)

LHS = gamma2.T.dot(FT.dot(gamma1)).real
RHS = FH.dot(gamma2).T.dot(gamma1).real

print("Self adjointness of FT: ", np.abs(LHS - RHS))

# Test that PSF convolution and M give the same answer
vec = np.ones(npix**2)
# vec = np.zeros([npix, npix])
# vec[npix//4:npix-npix//4, npix//4:npix-npix//4] = np.ones((npix//2+1)**2).reshape([npix//2+1, npix//2+1])
im_psf = PSF_probe(vec.flatten()).real
im_frrf = M(vec.flatten()).real

x = np.linspace(np.min(im_psf), np.max(im_psf), im_psf.size)
plt.figure('Difference between PSF convolution and M')
plt.plot(x, x, 'k')
plt.scatter(im_psf, im_frrf, marker='x')

########################################################################################################################

# plt.figure('Noise Covariance of DFT')
# plt.imshow(im_frrf.reshape([npix, npix]))
# plt.colorbar()

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
