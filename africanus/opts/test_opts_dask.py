# def test_pd_dask
from africanus.opts.primaldual import primal_dual_solver as pdd
from africanus.dft.dask import vis_to_im, im_to_vis
import dask.array as da
import numpy as np
import xarrayms
import matplotlib.pyplot as plt
from astropy.io import fits


def radec_to_lm(ra0, dec0, ra, dec):
    delta_ra = ra - ra0
    l = (np.cos(dec) * np.sin(delta_ra))
    m = (np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(delta_ra))
    return l, m


npix = 256

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

chunk = int(nrow/10)

# set up dask arrays
uvw_dask = da.from_array(uvw, chunks=(chunk, 3))
lm_dask = da.from_array(lm, chunks=(npix**2, 2))
frequency_dask = da.from_array(freq, chunks=nchan)
vis_dask = da.from_array(vis, chunks=(chunk, nchan))
weights_dask = da.from_array(weights, chunks=(chunk, nchan))

L_d = lambda image: im_to_vis(image, uvw_dask, lm_dask, frequency_dask).compute()
LT_d = lambda v: vis_to_im(v, uvw_dask, lm_dask, frequency_dask).compute()/wsum


PSF = vis_to_im(weights_dask, uvw_dask, lm_dask, frequency_dask).compute()


dirty = LT_d(vis_dask)

start = np.zeros_like(dirty)
start[int(npix**2/2), 0] = 10
start_dask = da.from_array(start, chunks=(npix**2))

cleaned = pdd(start_dask, vis_dask, L_d, LT_d, solver='spd', dask=True, maxiter=10)

plt.figure('ID Dask')
plt.imshow(dirty.reshape(npix, npix))
plt.colorbar()

plt.figure('IM Dask')
plt.imshow(cleaned.reshape(npix, npix))
plt.colorbar()

plt.show()

hdu = fits.PrimaryHDU(dirty.reshape(npix, npix))
hdul = fits.HDUList([hdu])
hdul.writeto('/home/antonio/Documents/Masters/Helpful_Stuff/WSCMSSSMFTestSuite/dirty_dask.fits', overwrite=True)
hdul.close()

hdu = fits.PrimaryHDU(cleaned.reshape(npix, npix))
hdul = fits.HDUList([hdu])
hdul.writeto('/home/antonio/Documents/Masters/Helpful_Stuff/WSCMSSSMFTestSuite/recovered_dask.fits', overwrite=True)
hdul.close()
