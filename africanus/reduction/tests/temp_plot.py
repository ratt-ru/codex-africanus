import dask.array as da
import numpy as np
from africanus.reduction.psf_redux import make_dim_reduce_ops, iFFT, FFT
from africanus.opts.data_reader import data_reader
import matplotlib.pyplot as plt

# generate frequencies
frequency = np.array([1.06e9])
ref_freq = 1
freq = frequency/ref_freq

data_path = "/home/antonio/Documents/Masters/Helpful_Stuff/WSCMSSSMFTestSuite/SSMF.MS_p0"
ra_dec = np.array([[3.15126500e-05], [-0.00551471375]])

uvw_dask, lm_dask, lm_pad_dask, frequency_dask, weights_dask, vis_dask, padding = data_reader(data_path, ra_dec, nrow=35100)

wsum = da.sum(weights_dask)
pad_pix = int(da.sqrt(lm_pad_dask.shape[0]))

try:
    vis_grid = np.fromfile('vis_grid.dat', dtype='complex128').reshape([pad_pix, pad_pix])
    PSF_hat = np.fromfile('psf_hat.dat', dtype='complex128').reshape([pad_pix, pad_pix])
except:
    vis_grid, PSF_hat, Sigma_hat = make_dim_reduce_ops(uvw_dask, lm_pad_dask, frequency_dask, vis_dask, weights_dask)
    vis_grid.tofile('vis_grid.dat')
    PSF_hat.tofile('psf_hat.dat')

uvw = uvw_dask.compute()

plt.figure('uv-coverage')
plt.plot(uvw[:, 0].flatten(), uvw[:, 1].flatten(), 'xb')
plt.plot(-uvw[:, 0].flatten(), -uvw[:, 1].flatten(), 'xb')

plt.figure('Gridded Weights')
plt.imshow(da.log(PSF_hat).real[padding:-padding, padding:-padding])

plt.figure('Dirty Image')
dirty = iFFT(vis_grid).real[padding:-padding, padding:-padding]
plt.imshow(dirty)

plt.figure("PSF")
PSF = FFT(PSF_hat).real[padding:-padding, padding:-padding]
plt.imshow(PSF)

plt.figure("Clean Image")
clean = np.zeros_like(dirty)
npix = pad_pix-2*padding
pos = np.argmax(dirty)
coords = [pos//npix, pos % npix]
clean[coords[0]-3:coords[0]+3 , coords[1]-3:coords[1]+3] = 100
plt.imshow(clean)

plt.show()
