from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dask.array as da
import numpy as np
import africanus.opts.tests.helpers_1d as h
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from africanus.opts.primaldual import primal_dual_solver

def FFT(x):
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x), norm='ortho'))


def iFFT(x):
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x), norm='ortho'))

np.random.seed(111)
da.random.seed(111)

NCPU = 8

sigma = 1.0

# generate u and l data
nrow = 1001

uvw = np.linspace(0.1, 10, nrow).reshape(nrow, 1)  # 1 + np.random.random(size=(nrow,1))*10  #
# uvw = 0.1 + 10 * np.random.random(size=(nrow, 1))

delta_l = 1.0/(3*uvw.max())
# npix = int(np.ceil(01.0/(uvw.min()*delta_l)))
# if npix % 2 == 0:
#     npix += 1

npix = 1001

lm_stop = delta_l*(npix//2)
lm = da.linspace(-lm_stop, lm_stop, npix, chunks=npix)
lm = lm.reshape([npix, 1])

uvw = da.from_array(uvw, chunks=[nrow//NCPU, 1])
weights = sigma**2 * da.ones_like(uvw, chunks=nrow//NCPU)
wsum = sum(weights)

# make matrix covariance
FFT_mat = h.make_fft_matrix(lm, npix)
iFFT_mat = FFT_mat.conj().T

Phi_mat = h.make_dft_matrix(uvw, lm, weights)
Phi_mat_adj = Phi_mat.conj().T/wsum

Inner_mat = Phi_mat_adj.dot(weights*Phi_mat)
Cov_mat = FFT_mat.dot(Phi_mat_adj.dot(weights*Phi_mat.dot(iFFT_mat)))

Cov_mat_adj = Cov_mat.conj().T

Inner_mat, Cov_mat, lm, uvw, weights, Phi_mat_adj, Phi_mat, wsum = da.compute(Inner_mat, Cov_mat, lm, uvw, weights, Phi_mat_adj, Phi_mat, wsum)

cov_diag = np.diagonal(Cov_mat)

weights = weights.flatten()

# plt.figure('FFT Real')
# plt.imshow(FFT_mat.real)
# plt.colorbar()
#
# plt.figure('FFT Imaginary')
# plt.imshow(FFT_mat.imag)
# plt.colorbar()
#
# plt.figure('Phi Real')
# plt.imshow(Phi_mat.real)
# plt.colorbar()
#
# plt.figure('Phi Imaginary')
# plt.imshow(Phi_mat.imag)
# plt.colorbar()
#
#
# plt.figure('RER')
# plt.imshow(Inner_mat.real)
# plt.colorbar()
#
# plt.figure("FRERF")
# plt.imshow(Cov_mat.real)
# plt.colorbar()

# make psf hat

psf = Phi_mat_adj.dot(weights)
# psf[psf<0] = 0

# plt.figure("PSF")
# plt.plot(lm, psf.real)

psf_hat = FFT_mat.dot(psf)*np.sqrt(wsum)
psf_hat[psf_hat<0] = 0

plt.figure("PSf_hat vs Diagonal")
plt.plot(lm, psf_hat, 'b')
plt.plot(lm, cov_diag.real, 'r')
plt.plot(lm, abs(psf_hat - cov_diag).real, 'g')

# make an image to convolve
nsource = 5
sources = np.random.randint(npix//4, 3*npix//4, nsource)
image = np.zeros_like(lm)
for pix in sources:
    image[pix] = np.random.random()*10

plt.figure('True image')
plt.plot(lm, image)

# transform = iFFT_mat.dot(FFT_mat.dot(image)).real
# transform = iFFT(FFT(image)).real
transform = Phi_mat_adj.dot(Phi_mat.dot(image)).real

plt.figure('Transformed image')
plt.plot(lm, transform)

plt.figure('difference')
plt.plot(lm, transform - image)

noise = (np.zeros(nrow) + 0.1*sigma*(np.random.randn(nrow) + 1.0j * np.random.randn(nrow))).flatten()

vis = Phi_mat.dot(image) # + noise.reshape([-1, 1])

# plt.figure("Visibility")
# plt.plot(uvw, vis.real)

dirty_image = Phi_mat_adj.dot(weights*vis.flatten()).real.reshape([-1, 1])
dirty_image[dirty_image<0] = 0

# plt.figure('Dirty image')
# plt.plot(lm, dirty_image)

grid_vis = FFT_mat.dot(dirty_image).reshape([-1, 1])

sigma_1 = 1/np.sqrt(psf_hat)
# sigma_1[np.isnan(sigma_1)] = 0

# plt.figure('Sigma Hat')
# plt.plot(lm, sigma_1)

white_vis = (sigma_1.flatten()*grid_vis.flatten()).reshape([-1, 1])

white_psf_hat = (sigma_1.flatten()*psf_hat.flatten()).reshape([-1, 1])

# plt.figure('white psf hat')
# plt.plot(lm, white_psf_hat.real)
# plt.plot(lm, white_psf_hat.imag, 'r')

# plt.figure('grid vis')
# plt.plot(lm, grid_vis.real)
# plt.plot(lm, grid_vis.imag, 'r')

# plt.figure('white vis')
# plt.plot(lm, white_vis.real)
# plt.plot(lm, white_vis.imag, 'r')

start = np.zeros([npix, 1])
start[npix//2] = 10

# operator = lambda img: (white_psf_hat.flatten()*h.FFT(img.flatten())).reshape([-1, 1])
# adjoint = lambda vis: h.iFFT(white_psf_hat.flatten()*vis.flatten()).real.reshape([-1, 1]) #/np.sqrt(sum(sigma_1))

# full_clean = primal_dual_solver(start, white_vis, operator, adjoint, maxiter=5000)

# operator = lambda img: (psf_hat.flatten()*FFT_mat.dot(img.flatten())).reshape([-1, 1])*np.sqrt(wsum)
# adjoint = lambda vis: iFFT_mat.dot(psf_hat.flatten()*vis.flatten()).reshape([-1, 1]).real
#
# full_clean = primal_dual_solver(start, grid_vis, operator, adjoint, tol=1e-8, maxiter=2000, solver='fbpd')

# operator = lambda img: (psf_hat.flatten()*FFT(img.flatten())).reshape([-1, 1])
# adjoint = lambda vis: iFFT((psf_hat.flatten()*vis.flatten()).reshape([-1, 1])).real/np.sqrt(wsum)
#
# full_clean = primal_dual_solver(start, grid_vis, operator, adjoint)

# full_op = lambda x: Cov_mat.dot(x)
# full_adj = lambda y: Cov_mat_adj.dot(y).real
#
# full_clean = primal_dual_solver(dirty_image, vis, full_op, full_adj, solver='rpd', dask=False, maxiter=2000, tol=1e-16).real

# full_op = lambda x: Phi_mat.dot(x)*np.sqrt(wsum)
# full_adj = lambda y: Phi_mat_adj.dot(y).real
#
# full_clean = primal_dual_solver(start, vis, full_op, full_adj, solver='fbpd', dask=False, maxiter=2000, tol=1e-16)

# plt.figure("Full Clean")
# plt.plot(lm, full_clean)
#
# print(full_clean.max()/image.max())
#
# plt.figure("Compare")
# plt.plot(lm, full_clean/(full_clean.max()/image.max())-image)

plt.show()
