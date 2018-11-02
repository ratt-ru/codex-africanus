from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from xarrayms import xds_from_ms, xds_from_table
import numpy as np
from africanus.gridding.util import estimate_cell_size
from africanus.constants import c as lightspeed
import matplotlib.pyplot as plt
import dask.array as da

_ARCSEC2RAD = np.deg2rad(1.0 / (60 * 60))


# rad/dec to lm coordinates (straight from fundamentals)
def radec_to_lm(ra0, dec0, ra, dec):
    delta_ra = ra - ra0
    l = (np.cos(dec) * np.sin(delta_ra))
    m = (np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(delta_ra))
    return l, m


def plot(im, stage, npix):
    plt.figure(stage)
    plt.imshow(im.reshape([npix, npix]).real)
    plt.colorbar()
    plt.show()


def gen_image_space(uvw, freqs, l_val, m_val):
    wavelengths = lightspeed / freqs
    cell_size = estimate_cell_size(uvw[:, 0], uvw[:, 1], wavelengths, factor=5).max()
    cell_size_rad = _ARCSEC2RAD * cell_size

    fov = max(max(abs(l_val)), max(abs(m_val))) * 1.5  # making sure the source is not at the edge of the field

    # set number of pixels required to properly oversample the image
    npix = int(2 * fov / cell_size_rad)
    if not npix % 2:
        npix += 1  # make sure it is odd

    print("You need to use at least npix = ", npix)

    x = np.linspace(-fov, fov, npix)
    cell_size = x[1] - x[0]  # might have changed slightly from the recommended value
    ll, mm = np.meshgrid(x, x)
    lm = np.vstack((ll.flatten(), mm.flatten())).T

    return lm, npix, cell_size, fov


def gen_padding_space(npix, pad_fact, cell_size, fov):

    padding = int(npix * pad_fact)
    pad_pix = npix + 2 * padding

    pad_range = fov + padding * cell_size
    x_pad = np.linspace(-pad_range, pad_range, pad_pix)
    ll_pad, mm_pad = np.meshgrid(x_pad, x_pad)
    lm_pad = np.vstack((ll_pad.flatten(), mm_pad.flatten())).T

    return lm_pad, pad_pix, padding


def data_reader(data_path, ra_dec=[0.1, 0.1], NCPU=8, nchan=1, nrow=1000, pad_fact=0.5):
    xds = list(xds_from_ms(data_path, chunks={"row": nrow}))[0]
    spw_ds = list(xds_from_table("::".join((data_path, "SPECTRAL_WINDOW")), group_cols="__row__"))[0]

    # make sure we are oversampling the image enough
    uvw = xds.UVW.data[0:nrow, :]
    # uvw[:, 2] = 0.0  # this is for testing
    vis = xds.DATA.data[0:nrow, 0:nchan, 0]
    freqs = spw_ds.CHAN_FREQ.data[0:nchan]

    # normalisation factor (equal to max(PSF))
    weights = xds.WEIGHT.data[0:nrow, 0:nchan]

    # generate lm-coordinates
    l_val, m_val = radec_to_lm(0, 0, ra_dec[0, :], ra_dec[1, :])
    print(l_val, m_val)

    lm, npix, cs, fov = gen_image_space(uvw, freqs, l_val, m_val)
    lm_pad, pad_pix, padding = gen_padding_space(npix, pad_fact, cs, fov)

    # Turn DFT into lambda functions for easy, single input access
    chunkrow = nrow // NCPU
    lm_dask = da.from_array(lm, chunks=(npix ** 2, 2))
    lm_pad_dask = da.from_array(lm_pad, chunks=(pad_pix ** 2, 2))
    uvw_dask = da.from_array(uvw, chunks=(chunkrow, 3))
    frequency_dask = spw_ds.CHAN_FREQ.data.rechunk(nchan)[0:nchan]
    weights_dask = da.from_array(weights, chunks=(chunkrow, nchan))
    vis_dask = da.from_array(vis, chunks=(chunkrow, nchan))

    # return all necessary information to the main process
    return uvw_dask, lm_dask, lm_pad_dask, frequency_dask, weights_dask, vis_dask, padding
