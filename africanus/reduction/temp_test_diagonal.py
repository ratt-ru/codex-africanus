import xarrayms
from africanus.dft.kernels import im_to_vis, vis_to_im
import matplotlib.pyplot as plt
from africanus.reduction.psf_redux import *


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
padding = [(npix**2 - npix)//2, (npix**2 - npix)//2]
PSF_pad = np.pad(PSF, padding, mode='constant')
PSF_hat = F(PSF_pad)

# Generate FFT and DFT matrices
R = np.zeros([nrow, npix**2], dtype='complex128')
F = np.zeros([npix**2, npix**2], dtype='complex128')
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
        F[u, v] += np.exp(-2j*np.pi*(j*l + k*m))/np.sqrt(F_norm)

# Generate adjoint matrices (conjugate transpose)
RH = R.conj().T
FH = F.conj().T


# Mostly here for reference, calculating the NC of the DFT using the operators and FFT matrix
def M(vec):
    T0 = FH.dot(vec).real.reshape([vec.shape[0], 1])
    T1 = L(T0)
    T2 = weights.dot(T1.T)
    T3 = LT(T2)
    T4 = F.dot(T3)
    T5 = T4.flatten()
    return T5


# Calculate NC matrix of
T0 = R.dot(FH)
w = np.diag(weights.flatten())
T1 = w.dot(T0)
T2 = RH.dot(T1)
M_mat = F.dot(T2).real
M_vec = np.diagonal(M_mat)


# the PSF probe: the commented code is what I was doing before, the uncommented is what I did today which yields much
# closer results to the main y=x line, though it still has a few zero values
def PSF_probe(vec):
    p = PSF_hat.dot(vec).real
    # T0 = FH.dot(vec)
    # T1 = F.dot(T0)
    # PSFH = F.dot(PSF.flatten())
    # T2 = PSFH*T1
    # T3 = FH.dot(T2)
    # T4 = F.dot(T3)
    # print('Method diff ', max(abs(T4-p)))
    return p


# doing the probing and calculating the PSF diagonal
D_vec = diag_probe(PSF_probe, npix**2).real  # not sure if should be taking abs or not, but I get a lot of negatives

# Set up y=x line
_min = min(min(D_vec), min(M_vec))
_max = max(max(D_vec), max(M_vec))
x = np.linspace(_min, _max, npix**2)

# plot diagonal comparison (lots of zeros in here when the probe uses PSF_hat)
plt.figure('Values of PSF NC vs FRRF NC')
plt.plot(x, x, 'k')
plt.scatter(M_vec, D_vec, marker='x')

# plot DFT NC
plt.figure('Noise Covariance of DFT')
plt.imshow(M_mat)
plt.colorbar()

# plot PSF NC from probing
plt.figure('Noise Covariance of PSF')
D = np.diag(D_vec)
plt.imshow(D)
plt.colorbar()

plt.show()
