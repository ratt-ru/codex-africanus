import xarrayms
from africanus.dft.dask import im_to_vis, vis_to_im
import dask.array as da
import matplotlib.pyplot as plt
from africanus.reduction.psf_redux import *


def radec_to_lm(ra0, dec0, ra, dec):
    delta_ra = ra - ra0
    l = (np.cos(dec) * np.sin(delta_ra))
    m = (np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(delta_ra))

    return l, m

npix = 30

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
    vis = ds.DATA.data.compute()[0:nrow, 0:nchan, 0]
    uvw = ds.UVW.data.compute()[0:nrow, :]
    weights = ds.WEIGHT.data.compute()[0:nrow, 0:nchan]

wsum = sum(weights)

chunk = nrow//10

uvw_dask = da.from_array(uvw, chunks=(chunk, 3))
lm_dask = da.from_array(lm, chunks=(npix**2, 2))
frequency_dask = da.from_array(freq, chunks=nchan)
vis_dask = da.from_array(vis, chunks=(chunk, nchan))
weights_dask = da.from_array(weights, chunks=(chunk, nchan))

L = lambda image: im_to_vis(image, uvw_dask, lm_dask, frequency_dask).compute()
LT = lambda v: vis_to_im(v, uvw_dask, lm_dask, frequency_dask).compute()/wsum

PSF = LT(weights_dask)

PSF = LT(weights).reshape([npix, npix])

dirty = np.random.randn(npix, npix)
start = np.random.randn(npix, npix)

padding = [(npix**2 - npix)//2, (npix**2 - npix)//2]
dirty = np.pad(dirty, padding, mode='constant')
start = np.pad(start, padding, mode='constant')
PSF_pad = np.pad(PSF, padding, mode='constant')
PSF_hat = F(PSF_pad)

# R = lambda v: PSF_response(v, PSF_hat)
# RH = lambda v: PSF_adjoint(v, PSF_hat)

# Rx = R(start).flatten()
# Ry = RH(dirty).flatten()
#
# LHS = dirty.flatten().T.dot(Rx)
# RHS = Ry.conj().T.dot(start.flatten())
#
# print(np.abs(LHS - RHS))

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

RH = R.conj().T
FH = F.conj().T
#
w = np.diag(weights.flatten())


def M(vec):
    T0 = FH.dot(vec).real.reshape([vec.shape[0], 1])
    T1 = L(T0)
    T2 = weights.dot(T1.T)
    T3 = LT(T2)
    T4 = F.dot(T3)
    T5 = T4.flatten()
    return T5


print(R.shape, FH.shape)
T0 = R.dot(FH)
T1 = w.dot(T0)
T2 = RH.dot(T1)
M_mat = F.dot(T2).real


def PSF_probe(vec):
    T0 = FH.dot(vec)
    T1 = F.dot(T0)
    PSFH = F.dot(PSF.flatten())
    T2 = PSFH*T1
    T3 = FH.dot(T2)
    T4 = F.dot(T3)
    return T4

# plt.figure('Noise Covariance of DFT Matrix')
# plt.imshow(covariance_mat)
# plt.colorbar()


D_vec = diag_probe(PSF_probe, npix**2).real
M_vec = np.diagonal(M_mat)
_min = min(min(D_vec), min(M_vec))
_max = max(max(D_vec), max(M_vec))
print(_min, _max)
x = np.linspace(_min, _max, npix**2)

plt.figure('Values of PSF NC vs FRRF NC')
plt.plot(x, x, 'k')
plt.scatter(M_vec, D_vec, marker='x')
# plt.figure('Noise Covariance of DFT')
# plt.imshow(M_mat)
# plt.colorbar()
#
# plt.figure('Noise Covariance of PSF')
# plt.imshow(D)
# plt.colorbar()
plt.show()
