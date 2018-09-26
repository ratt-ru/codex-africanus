import dask.array as da
import numpy as np
from africanus.dft.dask import im_to_vis, vis_to_im
import xarrayms
import matplotlib.pyplot as plt

npix = 512


def radec_to_lm(ra0, dec0, ra, dec):
    delta_ra = ra - ra0
    l = (np.cos(dec) * np.sin(delta_ra))
    m = (np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(delta_ra))
    return l, m


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

nrow = 10000
nchan = 1

for ds in xarrayms.xds_from_ms(data_path):
    Vdat = ds.DATA.data.compute()
    uvw = ds.UVW.data.compute()[0:nrow, :]
    weights = ds.WEIGHT.data.compute()[0:nrow, 0:nchan]

vis = Vdat[0:nrow, 0:nchan, 0]

wsum = sum(weights)

chunk = nrow//10

uvw_dask = da.from_array(uvw, chunks=(chunk, 3))
lm_dask = da.from_array(lm, chunks=(npix**2, 2))
frequency_dask = da.from_array(freq, chunks=nchan)
vis_dask = da.from_array(vis, chunks=(chunk, nchan))
weights_dask = da.from_array(weights, chunks=(chunk, nchan))

L = lambda image: im_to_vis(image, uvw_dask, lm_dask, frequency_dask).compute()
LT = lambda v: vis_to_im(v, uvw_dask, lm_dask, frequency_dask).compute()/wsum

PSF = LT(weights_dask).reshape([npix, npix])

plt.figure('uv-coverage')
plt.plot(uvw[:, 0].flatten(), uvw[:, 1].flatten(), 'xb')
plt.plot(-uvw[:, 0].flatten(), -uvw[:, 1].flatten(), 'xb')
plt.figure('PSF')
plt.imshow(PSF)

plt.show()
