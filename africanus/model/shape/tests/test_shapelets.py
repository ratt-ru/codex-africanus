import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from africanus.model.shape import shapelet, basis_function, shapelet_1d, shapelet_2d
from africanus.constants import c as lightspeed

import importlib.util
spec = importlib.util.spec_from_file_location("shapelets", "/home/vanstanden/shapelets/shapelets/shapelet.py")
shapelets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shapelets)
sl = shapelets

Fs = np.fft.fftshift
iFs = np.fft.ifftshift

from scipy import fftpack
fft = fftpack.fft
ifft = fftpack.ifft
fft2 = fftpack.fft2
ifft2 = fftpack.ifft2

def test_1d_shapelet():
	# set signal space coords
	beta = 1.0
	npix = 513
	coeffs = np.ones(1, dtype=np.float64)
	l_min = -15.0 * beta
	l_max = 15.0 * beta
	delta_l = (l_max - l_min)/(npix-1)
	if npix%2:
		l = l_min + np.arange(npix) * delta_l
	else:
		l = l_min + np.arange(-0.5, npix-0.5) * delta_l
	img_shape = shapelet_1d(l, coeffs, False, beta=beta)

	# get Fourier space coords and take fft
	u = Fs(np.fft.fftfreq(npix, d=delta_l))
	fft_shape = Fs(fft(iFs(img_shape)))

	# get uv space
	uv_shape = shapelet_1d(u, coeffs, True, delta_x=delta_l, beta=beta)

	assert np.allclose(uv_shape, fft_shape)


def test_2d_shapelet():
	# Define all respective values for nrow, ncoeff, etc
	beta = [.01, .01]
	nchan = 1
	ncoeffs = [1, 1]
	nsrc = 1

	# Define the range of uv values
	u_range = [-3 * np.sqrt(2) * (beta[0] ** (-1)), 3 * np.sqrt(2) * (beta[0] ** (-1))]
	v_range = [-3 * np.sqrt(2) * (beta[1] ** (-1)), 3 * np.sqrt(2) * (beta[1] ** (-1))]

	# Create an lm grid from the regular uv grid
	max_u = u_range[1]
	max_v = v_range[1]
	delta_x = 1/(2 * max_u) if max_u > max_v else 1/(2 * max_v)
	x_range = [-3 * np.sqrt(2) * beta[0], 3 * np.sqrt(2) * beta[0]]
	y_range = [-3 * np.sqrt(2) * beta[1], 3 * np.sqrt(2) * beta[1]]
	npix_x = int((x_range[1] - x_range[0]) / delta_x)
	npix_y = int((y_range[1] - y_range[0]) / delta_x)
	l_vals = np.linspace(x_range[0], x_range[1], npix_x)
	m_vals = np.linspace(y_range[0], y_range[1], npix_y)
	ll, mm = np.meshgrid(l_vals, m_vals)
	lm = np.vstack((ll.flatten(), mm.flatten())).T
	nrow = lm.shape[0]

	# Create input arrays
	img_coords = np.zeros((nrow, 3))
	img_coeffs = np.random.randn(nsrc, ncoeffs[0], ncoeffs[1])
	img_beta = np.zeros((nsrc, 2))
	frequency = np.empty((nchan), dtype=np.float)

	# Assign values to input arrays
	img_coords[:, :2], img_coords[:, 2] = lm[:,:], 0
	img_beta[0, :] = beta[:]
	frequency[:] = 1
	img_coeffs[:, :, :] = 1

	# Create output arrays
	gf_shapelets = np.zeros((nrow), dtype=np.float)

	ca_shapelets = shapelet_1d(img_coords[:,0], img_coeffs[0,0,:], False, beta=img_beta[0,0]) * shapelet_1d(img_coords[:,1], img_coeffs[0,0,:], False, beta=img_beta[0,1])

	for n1 in range(ncoeffs[0]):
			for n2 in range(ncoeffs[1]):
					c = img_coeffs[0,n1,n2]
					sl_dimensional_basis = sl.dimBasis2d(n1, n2, beta=beta)
					shapelets_basis_func = sl.computeBasis2d(sl_dimensional_basis, img_coords[:, 0], img_coords[:, 1])
					gf_shapelets[:] += c * shapelets_basis_func[:]
	# Compare griffinfoster (gf) shapelets to codex-africanus (ca) shapelets
	assert np.allclose(gf_shapelets, ca_shapelets)

def test_fourier_space_shapelets():
	# set overall scale
	beta_l = 1.0
	beta_m = 1.0
	
	# only taking the zeroth order with
	ncoeffs_l = 1
	ncoeffs_m = 1
	nsrc = 1
	coeffs_l = np.ones((nsrc, ncoeffs_l), dtype=np.float64)
	coeffs_m = np.ones((nsrc, ncoeffs_m), dtype=np.float64)
	
	# Define the range of lm values (these give 3 standard deviations for the 0th order shapelet in image space)
	scale_fact = 10.0
	l_min = -3 * np.sqrt(2) * beta_l * scale_fact
	l_max = 3 * np.sqrt(2) * beta_l * scale_fact
	m_min = -3 * np.sqrt(2) * beta_m * scale_fact
	m_max = 3 * np.sqrt(2) * beta_m * scale_fact
	
	# set number of pixels
	npix = 257

	# create image space coordinate grid
	delta_l = (l_max - l_min)/(npix-1)
	delta_m = (m_max - m_min)/(npix-1)
	lvals = l_min + np.arange(npix) * delta_l
	mvals = m_min + np.arange(npix) * delta_m
	assert lvals[-1] == l_max
	assert mvals[-1] == m_max
	ll, mm = np.meshgrid(lvals,mvals)
	lm = np.vstack((ll.flatten(),mm.flatten())).T


	img_space_shape = shapelet_1d(lm[:,0], coeffs_l[0,:], False, beta=beta_l) * shapelet_1d(lm[:,1], coeffs_m[0,:], False, beta=beta_m)

	# next take FFT
	fft_shapelet = Fs(fft2(iFs(img_space_shape.reshape(npix,npix))))
	fft_shapelet_max = fft_shapelet.real.max()
	fft_shapelet /= fft_shapelet_max

	# get freq space coords
	freq = Fs(np.fft.fftfreq(npix, d=(delta_l)))

	# Create uv grid
	uu, vv = np.meshgrid(freq, freq)
	nrows = uu.size
	assert nrows == npix**2
	uv = np.hstack((uu.reshape(nrows, 1), vv.reshape(nrows, 1)))
	uvw = np.zeros((nrows, 3), dtype=np.float64)
	uvw[:, 0:2] = uv

	# Define other parameters for shapelet call
	nchan = 1
	frequency = np.ones(nchan, dtype=np.float64) * lightspeed / (2 * np.pi)
	beta = np.zeros((nsrc, 2), dtype=np.float64)
	beta[0, 0] = beta_l
	beta[0, 1] = beta_m

	# Call the shapelet implementation
	coeffs_l = coeffs_l.reshape(coeffs_l.shape + (1,)) # We only have a single shape parameter, so we simply add another dimension onto it
	uv_space_shapelet = shapelet(uvw, frequency, coeffs_l, beta, (delta_l, delta_l)).reshape(npix, npix)
	uv_space_shapelet_max = uv_space_shapelet.real.max()
	uv_space_shapelet /= uv_space_shapelet_max

	assert np.allclose(fft_shapelet, uv_space_shapelet)



def test_dask_shapelets():
        da = pytest.importorskip('dask.array')
        from africanus.model.shape.dask import shapelet as da_shapelet
        from africanus.model.shape import shapelet as nb_shapelet

        row_chunks = (2,2)
        source_chunks = (5,10,5,5)

        row = sum(row_chunks)
        source = sum(source_chunks)
        nmax = [5, 5]
        beta_vals = [1., 1.]
        nchan=1

        np_coords = np.random.randn(row, 3)
        np_coeffs = np.random.randn(source, nmax[0], nmax[1])
        np_frequency = np.random.randn(nchan)
        np_beta = np.empty((source, 2))
        np_beta[:, 0], np_beta[:, 1] = beta_vals[0], beta_vals[1]
        np_delta_lm = np.array([1/(10 *np.max(np_coords[:,0])), 1/(10 * np.max(np_coords[:,1]))])

        da_coords = da.from_array(np_coords, chunks=(row_chunks, 3))
        da_coeffs = da.from_array(np_coeffs, chunks=(source_chunks, nmax[0], nmax[1]))
        da_frequency = da.from_array(np_frequency, chunks=(nchan,))
        da_beta = da.from_array(np_beta, chunks=(source_chunks, 2))
        delta_lm = da.from_array(np_delta_lm, chunks=(2))

        np_shapelets = nb_shapelet(np_coords,
                                np_frequency,
                                np_coeffs,
                                np_beta,
                                np_delta_lm)
        da_shapelets = da_shapelet(da_coords,da_frequency, da_coeffs, da_beta, delta_lm).compute()
        assert_array_almost_equal(da_shapelets, np_shapelets)
