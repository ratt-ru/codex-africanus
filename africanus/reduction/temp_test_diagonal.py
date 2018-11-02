import xarrayms
from africanus.dft.dask import im_to_vis, vis_to_im
import matplotlib.pyplot as plt
import numpy as np
from africanus.reduction.psf_redux import FFT, iFFT, PSF_response, PSF_adjoint, sigma_approx
import dask.array as da
from africanus.opts.data_reader import data_reader


data_path = "/home/antonio/Documents/Masters/Helpful_Stuff/WSCMSSSMFTestSuite/SSMF.MS_p0"
freq = da.array([1.06e9])
NCPU = 8

uvw_dask, lm_dask, lm_pad_dask, frequency_dask, weights_dask, vis_dask, padding = data_reader(data_path)

# normalisation factor (equal to max(PSF))
wsum = sum(weights_dask)
pad_pix = int(da.sqrt(lm_pad_dask.shape[0]))
npix = int(da.sqrt(lm_dask.shape[0]))

L = lambda image: im_to_vis(image, uvw_dask, lm_dask, frequency_dask).compute()
LT = lambda v: vis_to_im(v, uvw_dask, lm_dask, frequency_dask).compute()/np.sqrt(wsum)
L_pad = lambda image: im_to_vis(image, uvw_dask, lm_pad_dask, frequency_dask).compute()
LT_pad = lambda v: vis_to_im(v, uvw_dask, lm_pad_dask, frequency_dask).compute()/np.sqrt(wsum)
#
FT = np.fromfile('F.dat', dtype='complex128').reshape([pad_pix**2, pad_pix**2])
FH = FT.conj().T

# Generate the PSF using DFT
PSF = LT_pad(weights_dask).reshape([pad_pix, pad_pix])


# Mostly here for reference, calculatng the NC of the DFT using the operators and FFT matrix
def M(vec):
    T1 = L(da.from_array(vec.reshape([npix**2, 1]), chunks=(npix**2, 1)))
    T2 = weights_dask*T1
    T3 = LT(T2).real.reshape([npix, npix])
    return T3/np.sqrt(wsum)


def M_pad(vec):
    T1 = L_pad(da.from_array(vec.reshape([pad_pix**2, 1]), chunks=(pad_pix**2, 1)))
    T2 = weights_dask*T1
    T3 = LT_pad(T2).real.reshape([pad_pix, pad_pix])
    return T3.flatten()/np.sqrt(wsum)


PSF_hat = FFT(PSF)


# def PSF_probe(vec):
#     T0 = np.pad(vec.reshape([npix, npix]), [padding, padding], 'constant')
#     T1 = F(T0)
#     T2 = PSF_hat * T1
#     T3 = iF(T2).real
#     return T3[padding:-padding, padding:-padding].flatten()*np.sqrt(npix)  #pad_factor#*(pad_pix - npix)/pad_pix


sigma = np.ones(pad_pix**2)
P = lambda image: PSF_response(image, PSF_hat, sigma)*np.sqrt(pad_pix**2/wsum)
PH = lambda image: PSF_adjoint(image, PSF_hat, sigma)*np.sqrt(pad_pix**2/wsum)

########################################################################################################################
# # Test that RH and LT_pad produce the same value
# test_ones = np.ones_like(weights)
# test0 = RH.dot(test_ones).real
# test1 = LT_pad(test_ones).real
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
#
# # Test self adjointness of FT FH
# gamma1 = np.random.randn(pad_pix**2)
# gamma2 = np.random.randn(pad_pix**2)
#
# LHS = gamma2.T.dot(FT.dot(gamma1)).real
# RHS = FH.dot(gamma2).T.dot(gamma1).real
#
# print("Self adjointness of FT: ", np.abs(LHS - RHS))


########################################################################################################################
# Test that PSF convolution and M give the same answer
# vec = np.random.random(npix**2).reshape([npix, npix])
# vec = np.ones([npix, npix])
vec = np.zeros([npix, npix])
vec[npix//2, npix//2] = 1
im_psf = iFFT(P(np.pad(vec, padding, 'constant'))).real[padding:-padding, padding:-padding].flatten()
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

plt.figure('Fourier difference')
F_temp = FFT(np.zeros([pad_pix**2,pad_pix**2]))
test = np.eye(pad_pix**2)
plt.imshow(np.abs(FT.dot(test) - FFT(test)))
plt.colorbar()

# # doing the probing and calculating the PSF diagonal
D_vec = sigma_approx(PSF)
print(D_vec.shape)
M_mat = FT.dot(LT_pad(L_pad(FH))).real/np.sqrt(pad_pix*npix)
# # M_mat = RH.dot(w.dot(R)).real
# # M_mat = diag_probe(M_pad, pad_pix**2)

# reshaping and removing padding from D_vec created by the PSF
M_vec = np.diagonal(M_mat)
print(M_vec.shape)

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

# plot PSF NC from probing
plt.figure('Noise Covariance of PSF')
D = np.diag(D_vec)
plt.imshow(D)
plt.colorbar()
#
plt.show()
