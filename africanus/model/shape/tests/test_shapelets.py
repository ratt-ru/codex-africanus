import numpy as np
from numpy.testing import assert_array_almost_equal
#import pytest
from africanus.model.shape.shapelets import shapelet as sl
from scipy.special import factorial, hermite
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pytest
from africanus.model.shape import shapelet, basis_function, shapelet_1d, shapelet_2d
from africanus.constants import c as lightspeed
from dask.diagnostics import ProgressBar
import pickle
# from shapelets import *

Fs = np.fft.fftshift
iFs = np.fft.ifftshift

from scipy import fftpack
fft = fftpack.fft
ifft = fftpack.ifft
fft2 = fftpack.fft2
ifft2 = fftpack.ifft2

def _test_N6251_vals():
        da = pytest.importorskip("dask.array")
        from africanus.model.shape.dask import shapelet as ca_shapelet

        p_file = open("./N6251_cart.pkl", "rb")
        d = pickle.load(p_file, encoding='latin1')#['coeffs'].reshape((1, 16,16))[:,:npoly, :npoly]

        d_size = (83,107)
        xc = [52., 49.]
        beta_vals = [2.5, 6.0]
        npoly=16

        ry=np.array(range(0,d['size'][0]),dtype=float)-d['xc'][0]
        rx=np.array(range(0,d['size'][1]),dtype=float)-d['xc'][1]

        yy=np.reshape(np.tile(ry,len(rx)),(len(rx),len(ry))).T
        xx=np.reshape(np.tile(rx,len(ry)),(len(ry),len(rx)))

        gf_shapelets = np.load("./shapelet_out.npz")['mdl']

        xy = np.vstack((xx.flatten(), yy.flatten())).T
        plt.figure("X coords")
        plt.imshow(xx)
        plt.colorbar()
        plt.savefig("./x_coords.png")
        plt.close()
        plt.figure("Y coords")
        plt.imshow(yy)
        plt.colorbar()
        plt.savefig("./y_coords.png")
        plt.close()

        ca_coords = np.zeros((xy.shape[0], 3))
        ca_coords[:,:2] = xy
        ca_coeffs = d['coeffs'].reshape((1, 16,16))[:,:npoly, :npoly]
        ca_beta = np.array(d['beta']).reshape((1,2))
        ca_frequency = np.array([lightspeed / (2 * np.pi)])
        # ca_frequency = np.array([1e06])
        # delta_lm = np.array([1/(10 * np.max(ca_coords[:,0])), 1/(10 * np.max(ca_coords[:,1]))])
        delta_lm = np.array([(ca_coords[1,0] - ca_coords[0,0])/1000, (ca_coords[d['size'][1],1] - ca_coords[0,1])/1000])
        print(delta_lm)
        # print("coords shape is : ",ca_coords.shape)
        # quit()

        row_chunks = ca_coords.shape[0] // 5

        print("calling incorrect shapelet with coordinate range ",np.max(ca_coords[:,0]) - np.min(ca_coords[:,0]), np.max(ca_coords[:,1]) - np.min(ca_coords[:,1])," frequency ", ca_frequency, " \n coeffs ", ca_coeffs, "\n beta ", ca_beta, "\n and deltas ", delta_lm)

        c_computations = ca_shapelet(da.from_array(ca_coords, chunks=(row_chunks, 3)),
                        da.from_array(ca_frequency, chunks=ca_frequency.shape),
                        da.from_array(ca_coeffs, chunks=ca_coeffs.shape),
                        da.from_array(ca_beta[:,::-1], chunks=ca_beta.shape),
                        da.from_array(delta_lm, chunks=delta_lm.shape))
        c_computations = c_computations / np.max(np.abs(c_computations))
        c = None
        with ProgressBar():
                c = da.compute(c_computations)[0][:,0,0].reshape(gf_shapelets.shape)


        plt.figure('Codex Africanus')
        plt.imshow(c.real)
        plt.colorbar()
        plt.savefig("Codex_N6251.png")
        plt.close()

        plt.figure('Griffinfoster')
        plt.imshow(gf_shapelets.real)
        plt.colorbar()
        plt.savefig("Griffin_N6251.png")
        plt.close()

        # print(yy.shape, xx.shape, gf_shapelets.shape, c[0][:,0,0].reshape(gf_shapelets.shape).shape)
        
        

def _test_N6251():
        da = pytest.importorskip('dask.array')

        from africanus.model.shape.dask import shapelet
        npix = 128
        row_chunks = npix // 8
        source_chunks = (1,)
        npoly=15

        row = npix**2
        source = sum(source_chunks)
        beta_vals = [2.5, 6.0]
        nchan=1 

        scale_fact = 4
        l_min = -3 * np.sqrt(2) * beta_vals[0] * scale_fact
        l_max = 3 * np.sqrt(2) * beta_vals[0] * scale_fact
        m_min = l_min#-3 * np.sqrt(2) * beta_vals[1] * scale_fact
        m_max = l_max#3 * np.sqrt(2) * beta_vals[1] * scale_fact
        delta_l = (l_max - l_min)/((npix-1) * 10)

        l = np.linspace(l_min, l_max, npix)
        m=np.linspace(m_min, m_max, npix)

        mm, ll = np.meshgrid(l,m)

        lm=np.vstack((ll.flatten(), mm.flatten())).T
        plt.figure()
        plt.imshow(mm)
        plt.colorbar()
        plt.savefig("./ll.png")
        plt.close()

        freq = Fs(np.fft.fftfreq(npix, d=delta_l))
        uu, vv = np.meshgrid(freq, freq)
        nrows = uu.size
        assert nrows == npix**2
        uv = np.hstack((uu.reshape(nrows, 1), vv.reshape(nrows, 1)))

        p_file = open("./N6251_cart.pkl", "rb")
        p_coeffs = pickle.load(p_file, encoding='latin1')['coeffs'].reshape((1, 16,16))[:,:npoly, :npoly]

        s_coeffs = np.load("./shapelet_coeffs.npz")['coeffs'][:,:npoly,:npoly]
        # d=np.load("./shapelet_coeffs.npz")
        print("coeffs shape is",s_coeffs.shape)
        nmax = [s_coeffs.shape[1], s_coeffs.shape[2]]

        # ry=np.array(range(0,npix),dtype=float)-52.0
        # rx=np.array(range(0,npix),dtype=float)-49.0

        # yy=np.reshape(np.tile(ry,len(rx)),(len(rx),len(ry)))
        # xx=np.reshape(np.tile(rx,len(ry)),(len(ry),len(rx)))

        # xy = np.vstack((xx.flatten(), yy.flatten())).T

        # print(xy.shape)
        # quit()   

        np_coords = np.empty((row, 3))
        np_coords[:, :2], np_coords[:,2] = lm, 0
        np_coeffs = np.random.randn(source, nmax[0], nmax[1])
        np_coeffs[...] = p_coeffs[...]
        np_frequency = np.ones((nchan,)) * (lightspeed / (2*np.pi))
        np_beta = np.empty((source, 2))
        np_beta[:, 0], np_beta[:, 1] = beta_vals[0], beta_vals[1]
        np_delta_lm = np.array([1/(10 *np.max(np_coords[:,0])), 1/(10 * np.max(np_coords[:,1]))])

        print("Min coordinate : ", np.min(np_coords))


        da_coords = da.from_array(np_coords, chunks=(row_chunks, 3))
        da_coeffs = da.from_array(np_coeffs, chunks=(source_chunks, nmax[0], nmax[1]))
        da_frequency = da.from_array(np_frequency, chunks=(nchan,))
        da_beta = da.from_array(np_beta, chunks=(source_chunks, 2))
        delta_lm = da.from_array(np_delta_lm, chunks=(2))


        print("calling correct shapelet with coordinate range ",np.max(np_coords[:,0]) - np.min(np_coords[:,0]), np.max(np_coords[:,1]) - np.min(np_coords[:,1])," frequency ", np_frequency, " \n coeffs ", np_coeffs, "\n beta ", np_beta, "\n and deltas ", np_delta_lm)

        da_shapelets = shapelet(da_coords,da_frequency, da_coeffs, da_beta, delta_lm)
        print(da_shapelets.shape)
        sl_vals = None
        with ProgressBar():
                # sl_vals = ifft2(iFs(da_shapelets[:,0,0].compute().reshape((npix,npix)))).real
                sl_vals = da_shapelets[:,0,0].compute().reshape((npix,npix))
        print(sl_vals[np.where(sl_vals > 0)])
        print(np.allclose(sl_vals, sl_vals[0,0]))
        print(np.max(sl_vals) - np.min(sl_vals))
        sl_vals = sl_vals / np.max(np.abs(sl_vals))
        plt.figure('N6251')
        plt.imshow(np.abs(sl_vals))
        plt.colorbar()
        # plt.show()
        plt.savefig("./N6251.png")
        plt.close()


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
    fft_shape_max = fft_shape.real.max()

    # get uv space
    uv_shape = shapelet_1d(u, coeffs, True, delta_x=delta_l, beta=beta)
    uv_shape_max = uv_shape.real.max()

    print('ratio = ', uv_shape_max/fft_shape_max)

    plt.figure('uv')
    plt.plot(uv_shape.real, 'k')

    plt.figure('fft')
    plt.plot(fft_shape.real, 'k')

    plt.figure('diff')
    plt.plot(uv_shape.real - fft_shape.real, 'k')


    plt.show()

def _test_shapelets_against_gaussian():
        from africanus.model.shape import gaussian
        from africanus.model.shape import shapelet
        beta=np.array([[1.0, 1.0]], dtype=np.float64)
        npix=33
        ncoeffs_l = 1
        ncoeffs_m = 1
        coeffs_l=np.ones((1, ncoeffs_l, ncoeffs_l), dtype=np.float64)
        coeffs_m = np.ones((1, ncoeffs_m, ncoeffs_m), dtype=np.float64)
        l_min = -15.0 * beta[0, 0]
        l_max = 15.0 * beta[0, 0]
        m_min = -15.0 * beta[0, 1]
        m_max = 15.0 * beta[0, 1]
        delta_l = (l_max - l_min) / (npix - 1)
        delta_m = (m_max - m_min) / (npix - 1)
        if npix%2:
                l = l_min + np.arange(npix) * delta_l
                m = m_min + np.arange(npix) * delta_m
        else:
                l = l_min + np.arange(npix - 1) * delta_l
                m = m_min + np.arange(npix - 1) * delta_m
        source_center = np.array([[l[len(l) // 2], m[len(m) // 2]]])
        print(source_center)
        
        u = Fs(np.fft.fftfreq(npix, d=delta_l))
        v = Fs(np.fft.fftfreq(npix, d=delta_m))
        w = Fs(np.fft.fftfreq(npix, d=np.sqrt(delta_l**2 + delta_m**2)))
        w[:] = 0
        uu, vv = np.meshgrid(u, v)
        uvw_vstack = np.vstack((uu.flatten(), vv.flatten()))
        uvw = np.empty((npix ** 2, 3))
        uvw[:, :2] = uvw_vstack.T
        uvw[:, 2] = 0
        print("uvw is ", uvw)

        frequency_shapelets=np.array([lightspeed * (np.sqrt(2))/ (4 * np.pi**2)], dtype=np.float64)
        frequency_gaussian = np.array([(lightspeed * np.sqrt(np.log(256))) / (np.sqrt(2) * np.pi)], dtype=np.float64)
        gaussian_params=np.array([[1., 1., 0.]], dtype=np.float64)

        print("starting shapelets now")
        print(uvw.shape)
        print(frequency_shapelets.shape)
        print(coeffs_l)
        print(coeffs_m.shape)
        print(beta.shape)
        print(delta_l)
        uv_shape = shapelet(uvw, 
                frequency_shapelets, 
                coeffs_l, 
                beta, 
                np.array([delta_l, delta_m]))[:,0,0] #* (delta_l * np.sqrt(np.sqrt(np.pi)) / np.sqrt(2 * np.pi) ) * (delta_m * np.sqrt(np.sqrt(np.pi)) / np.sqrt(2 * np.pi) )# * (delta_l * delta_m *np.sqrt(np.pi)) / (2 * np.pi) #shapelet_2d(u, v, coeffs_l, coeffs_m, True, delta_x=delta_l, delta_y=delta_m, beta=beta)
        print("finished shapelets")
        gaussian_shape = gaussian(uvw, frequency_gaussian, gaussian_params)[0,:,0]
        print("gaussian shape = ", gaussian_shape.shape)

        uv_shape_max = np.abs(uv_shape.real).max()
        print("uv_shape_max = ", uv_shape_max)
        gaussian_shape_max = np.abs(gaussian_shape).max()
        print("gaussian_shape_max = ", gaussian_shape_max)

        uv_shape = uv_shape / uv_shape_max
        gaussian_shape = gaussian_shape / gaussian_shape_max
        
        print("maximum Shapelet, maximum Gauss = ", uv_shape_max, gaussian_shape_max)
        print("ratio = ", uv_shape_max / gaussian_shape_max)
        print("shape = ", np.allclose(uv_shape.real, gaussian_shape))
        print("max diff = ", np.max(uv_shape.real - gaussian_shape))

        plt.figure('Shapelet')
        plt.imshow(uv_shape.real.reshape(npix, npix))
        plt.colorbar()
        plt.savefig("./shapelet.png")
        plt.close()

        plt.figure('Gaussian')
        plt.imshow(gaussian_shape.reshape(npix, npix))
        plt.colorbar()
        plt.savefig("./gaussian.png")
        plt.close()

        plt.figure('Difference')
        plt.imshow((uv_shape.real - gaussian_shape).reshape(npix, npix))
        plt.colorbar()
        plt.savefig("./difference.png")
        plt.close()
        # plt.show()



def test_2d_shapelet():
	beta=1.0
	npix=129
	ncoeffs_l = 1
	ncoeffs_m = 1
	coeffs_l=np.ones((ncoeffs_l, ncoeffs_m), dtype=np.float64)
	coeffs_m = np.ones(ncoeffs_m, dtype=np.float64)
	l_min = -15.0 * beta
	l_max = 15.0 * beta
	m_min = -15.0 * beta
	m_max = 15.0 * beta
	delta_l = (l_max - l_min) / (npix - 1)
	delta_m = (m_max - m_min) / (npix - 1)
	if npix%2:
		l = l_min + np.arange(npix) * delta_l
		m = m_min + np.arange(npix) * delta_m
	else:
		l = l_min + np.arange(npix - 1) * delta_l
		m = m_min + np.arange(npix - 1) * delta_m
	img_shape = shapelet_2d(l, m, coeffs_l, coeffs_m, False, beta=beta)

	u = Fs(np.fft.fftfreq(npix, d=delta_l))
	v = Fs(np.fft.fftfreq(npix, d=delta_m))
	fft_shape = Fs(fft2(iFs(img_shape)))
	fft_shape_max = fft_shape.real.max()

	uv_shape = shapelet_2d(u, v, coeffs_l, True, delta_x=delta_l, delta_y=delta_m, beta=beta)
	uv_shape_max = uv_shape.real.max()

	print("ratio = ", uv_shape_max / fft_shape_max)

	plt.figure('uv')
	plt.imshow(uv_shape.real)
	plt.colorbar()

	plt.figure('fft')
	plt.imshow(fft_shape.real)
	plt.colorbar()

	plt.figure('diff')
	plt.imshow(fft_shape.real - uv_shape.real)
	plt.colorbar()

	plt.show()

def _test_2d_shapelet():
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

        # Create regular uv grid
        freqs_u = Fs(np.fft.fftfreq(npix_x, d=delta_x))
        freqs_v = Fs(np.fft.fftfreq(npix_y, d=delta_x))
        uu, vv = np.meshgrid(freqs_u, freqs_v)
        uv = np.vstack((uu.flatten(), vv.flatten())).T

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
        pt_shapelets = np.zeros((nrow))

        for row in range(nrow):
                pt_tmp_shapelet = 0
                l, m = img_coords[row, :2]
                for n1 in range(ncoeffs[0]):
                        for n2 in range(ncoeffs[1]):
                                c = img_coeffs[0, n1, n2]
                                pt_basis_func = shapelet_img_space(l, n1, beta[0]) * shapelet_img_space(m, n2, beta[1])
                                pt_tmp_shapelet += c * pt_basis_func
                pt_shapelets[row] = pt_tmp_shapelet

        for n1 in range(ncoeffs[0]):
                for n2 in range(ncoeffs[1]):
                        sl_dimensional_basis = sl.dimBasis2d(n1, n2, beta=beta)
                        shapelets_basis_func = sl.computeBasis2d(sl_dimensional_basis, img_coords[:, 0], img_coords[:, 1])
                        gf_shapelets[:] += c * shapelets_basis_func[:]

        assert np.allclose(gf_shapelets, pt_shapelets)



def test_image_space():
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

        # Create regular uv grid
        freqs_u = Fs(np.fft.fftfreq(npix_x, d=delta_x))
        freqs_v = Fs(np.fft.fftfreq(npix_y, d=delta_x))
        uu, vv = np.meshgrid(freqs_u, freqs_v)
        uv = np.vstack((uu.flatten(), vv.flatten())).T

        ###############################################################
        ################ BEGIN IMAGE SPACE TEST (passes) ##############
        ###############################################################
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
        pt_shapelets = np.zeros((nrow))
        """
        for n1 in range(ncoeffs[0]):
                for n2 in range(ncoeffs[1]):
                        sl_dimensional_basis = sl.dimBasis2d(n1, n2, beta=beta)
                        c = img_coeffs[0, n1, n2]
                        pt_basis_func = shapelet_img_space(img_coords[:, 0], n1, beta[0]) * shapelet_img_space(img_coords[:, 1], n2, beta[1])
                        shapelets_basis_func = sl.computeBasis2d(sl_dimensional_basis, img_coords[:, 0], img_coords[:, 1])
                        gf_shapelets[:] += c * shapelets_basis_func
                        pt_shapelets[:] += c * pt_basis_func
        """
        for row in range(nrow):
                pt_tmp_shapelet = 0
                l, m = img_coords[row, :2]
                for n1 in range(ncoeffs[0]):
                        for n2 in range(ncoeffs[1]):
                                c = img_coeffs[0, n1, n2]
                                pt_basis_func = shapelet_img_space(l, n1, beta[0]) * shapelet_img_space(m, n2, beta[1])
                                pt_tmp_shapelet += c * pt_basis_func
                pt_shapelets[row] = pt_tmp_shapelet

        for n1 in range(ncoeffs[0]):
                for n2 in range(ncoeffs[1]):
                        sl_dimensional_basis = sl.dimBasis2d(n1, n2, beta=beta)
                        shapelets_basis_func = sl.computeBasis2d(sl_dimensional_basis, img_coords[:, 0], img_coords[:, 1])
                        gf_shapelets[:] += c * shapelets_basis_func[:]



        plt.figure()
        plt.imshow(pt_shapelets.reshape((npix_x, npix_y)))
        plt.colorbar()
        plt.title("Image Space Shapelets")
        #plt.show()
        plt.savefig("./img_shapelets.png")
        plt.close()

        plt.figure()
        plt.imshow(gf_shapelets.reshape((npix_x, npix_y)))
        plt.colorbar()
        plt.title("Griffinfoster Shapelets")
        #plt.show()
        plt.savefig("./gf_shapelets.png")
        plt.close()

        assert np.allclose(gf_shapelets, pt_shapelets)
        ###############################################################
        ################# END IMAGE SPACE TEST ########################
        ###############################################################

        ###############################################################
        ################# BEGIN FOURIER SPACE TEST ####################
        ###############################################################
        # Call shapelet script
        f_coords = np.zeros((nrow, 3))
        f_frequencies = np.empty(nchan)
        f_coeffs = img_coeffs
        f_beta = img_beta

        f_coords[:, :2], f_coords[:, 2] = uv[:, :], 0
        f_frequencies[:] = 1
        print(uv)
        print("running numba now")
        ca_shapelets = nb_shapelet(f_coords, f_frequencies, f_coeffs, f_beta)
        print("numba finished running")
        """
        pt_shapelets = np.zeros((nrow))
        for n1 in range(ncoeffs[0]):
                for n2 in range(ncoeffs[1]):
                        c = f_coeffs[0, n1, n2]
                        pt_ft_img_func = shapelet_img_space(f_coords[:, 0], n1, f_beta[0, 0]) * shapelet_img_space(f_coords[:, 1], n2, f_beta[0, 1])
                        pt_shapelets += c * pt_ft_img_func
        """
        ft_shapelets = Fs(np.fft.fft2(pt_shapelets.reshape((npix_x,npix_y))))
        ca_shapelets = ca_shapelets[0, :,  0]

        ca_shapelets = np.abs(ca_shapelets.reshape((npix_x, npix_y)))
        ft_shapelets = np.abs(ft_shapelets)

        #print(np.average(ft_shapelets / ca_shapelets))
        print("******************************")



        plt.figure()
        plt.imshow(ft_shapelets)
        plt.colorbar()
        plt.title("FFT Shapelets")
        plt.show()
        plt.savefig("./ft_shapelets.png")
        plt.close()

        plt.figure()
        plt.imshow(ca_shapelets)
        plt.colorbar()
        plt.title("Codex Africanus Shapelets")
        plt.show()
        plt.savefig("./ca_shapelets.png")
        plt.close()


        plt.figure()
        plt.imshow(ca_shapelets / ft_shapelets)
        plt.colorbar()
        plt.title("Quotient")
        plt.show()
        plt.savefig("./division.png")
        plt.close()


        plt.figure()
        plt.imshow(ca_shapelets - ft_shapelets)
        plt.colorbar()
        plt.title("Difference")
        plt.show()
        plt.savefig("./difference.png")
        plt.close()


        assert np.allclose(ca_shapelets, ft_shapelets)
        #####################################################
        #####################################################
        #####################################################

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
    scale_fact = 10
    l_min = -3 * np.sqrt(2) * beta_l * scale_fact
    l_max = 3 * np.sqrt(2) * beta_l * scale_fact
    m_min = -3 * np.sqrt(2) * beta_m * scale_fact
    m_max = 3 * np.sqrt(2) * beta_m * scale_fact
    # set number of pixels
    npix = 513
    # create image space coordinate grid
    delta_l = (l_max - l_min)/(npix-1)
    delta_m = (m_max - m_min)/(npix-1)
    lvals = l_min + np.arange(npix) * delta_l
    mvals = m_min + np.arange(npix) * delta_m
    assert lvals[-1] == l_max
    assert mvals[-1] == m_max
    # evaluate image space shapelet
    img_space_shape = np.zeros((npix, npix), dtype=np.float64)
    for i, l in enumerate(lvals):
        for j, m in enumerate(mvals):
            for nx in range(ncoeffs_l):
                for ny in range(ncoeffs_m):
                    img_space_shape[i, j] += coeffs_l[0, nx] * basis_function(nx, l, beta_l) \
                                             * coeffs_m[0, ny] * basis_function(ny, m, beta_m)

    # ll, mm = np.meshgrid(lvals, mvals)
    # img_space_shape_test = np.exp(-ll**2/(2*beta_l**2))*np.exp(-mm**2/(2*beta_m**2))/(np.sqrt(np.sqrt(np.pi)*beta_l)*np.sqrt(np.sqrt(np.pi)*beta_m))

    # next take FFT
    fft_shapelet = Fs(fft(iFs(img_space_shape)))
    print("FFT imag", np.abs(fft_shapelet.imag).max())
    fft_shapelet_max = fft_shapelet.real.max()
    fft_shapelet /= fft_shapelet_max

    # get freq space coords
    freq = Fs(np.fft.fftfreq(npix, d=delta_l))
    uu, vv = np.meshgrid(freq, freq)
    nrows = uu.size
    assert nrows == npix**2
    uv = np.hstack((uu.reshape(nrows, 1), vv.reshape(nrows, 1)))
    uvw = np.zeros((nrows, 3), dtype=np.float64)
    uvw[:, 0:2] = uv
    nchan = 1
    frequency = np.ones(nchan, dtype=np.float64) * lightspeed
    beta = np.zeros((nsrc, 2), dtype=np.float64)
    beta[0, 0] = beta_l
    beta[0, 1] = beta_m

    uv_space_shapelet = shapelet(uvw, frequency, coeffs_l, coeffs_m, beta).reshape(npix, npix)
    print("uv imag = ", np.abs(uv_space_shapelet.imag).max())
    uv_space_shapelet_max = uv_space_shapelet.real.max()
    uv_space_shapelet /= uv_space_shapelet_max
    # plt.figure('uv')
    # plt.imshow(uv_space_shapelet.real)
    # plt.colorbar()
    # plt.figure('fft')
    # plt.imshow(fft_shapelet.real)
    # plt.colorbar()
    # plt.figure('diff')
    # plt.imshow(fft_shapelet.real - uv_space_shapelet.real)
    # plt.colorbar()
    # plt.show()
    ratio = fft_shapelet_max/uv_space_shapelet_max
    print(" npix ", (1.0/np.sqrt(2*np.pi * npix)) / ratio,\
                1.0/(2*np.pi)/ ratio,\
                1.0/np.sqrt(2*np.pi)**3  / ratio, \
                1.0/(2*np.pi)**(2) / ratio, \
                np.sqrt(np.pi) * delta_l*delta_m / ratio, \
                1/npix / ratio)
    print(" Ratio = ", ratio, 1.0/ratio)
    print(" Max diff = ", np.abs(fft_shapelet - uv_space_shapelet).max())

def _test_shapelet():
    npix = 15
    nrow = npix **2
    nsrc = 1
    nmax = [1,1]
    beta_vals = [1., 1.]

    u_range = [-3 * np.sqrt(2) *(beta_vals[0] ** (-1)), 3 * np.sqrt(2) * (beta_vals[0] ** (-1))]
    v_range = [-3 * np.sqrt(2) *(beta_vals[1] ** (-1)), 3 * np.sqrt(2) * (beta_vals[1] ** (-1))]

    du = (u_range[1] - u_range[0]) / npix
    dv = (v_range[1] - v_range[0]) / npix
    freqs_u = Fs(np.fft.fftfreq(npix, d=du))
    freqs_v = Fs(np.fft.fftfreq(npix, d=dv))
    uu, vv = np.meshgrid(freqs_u, freqs_v)
    uv = np.vstack((uu.flatten(), vv.flatten())).T

    coords = np.empty((nrow, 3))
    coeffs = np.empty((nsrc, nmax[0], nmax[1]))
    beta = np.empty((nsrc, 2))

    coords[:, :2], coords[:, 2] = uv, 0
    coeffs[0, :, :] = np.random.randn(nmax[0], nmax[1])
    beta[0, 0], beta[0, 1] = beta_vals[0], beta_vals[1]

    out_shapelet = nb_shapelet(coords, coeffs, beta)

    assert True

def _test_shapelet_vals():
    npix = 35
    nrow = npix **2
    nsrc = 1
    nmax = [10, 10]
    beta_vals = [1., 1.]

    u_range = [-3 * np.sqrt(2) *(beta_vals[0] ** (-1)), 3 * np.sqrt(2) * (beta_vals[0] ** (-1))]
    v_range = [-3 * np.sqrt(2) *(beta_vals[1] ** (-1)), 3 * np.sqrt(2) * (beta_vals[1] ** (-1))]

    du = (u_range[1] - u_range[0]) / npix
    dv = (v_range[1] - v_range[0]) / npix
    freqs_u = Fs(np.fft.fftfreq(npix, d=du))
    freqs_v = Fs(np.fft.fftfreq(npix, d=dv))
    uu, vv = np.meshgrid(freqs_u, freqs_v)
    uv = np.vstack((uu.flatten(), vv.flatten())).T


    coords = np.empty((nrow, 3), dtype=np.float)
    coeffs = np.empty((nsrc, nmax[0], nmax[1]), dtype=np.float)
    beta = np.empty((nsrc, 2), dtype=np.float)

    coords[:, :2], coords[:, 2] = uv, 0
    coeffs[0, :, :] = np.random.randn(nmax[0], nmax[1])
    #coeffs[0, :, :] = 4.
    beta[0, 0], beta[0, 1] = beta_vals[0], beta_vals[1]

    codex_shapelets = nb_shapelet(coords, coeffs, beta).reshape((npix, npix))

    gf_shapelets = np.zeros((nrow), dtype=np.complex128)
    for n1 in range(nmax[0]):
        for n2 in range(nmax[1]):
            gf_shapelets += coeffs[0, n1, n2] * sl.shapelet.computeBasis2d(sl.shapelet.dimBasis2d(n1, n2, beta=beta_vals, fourier=True), uv[:, 0], uv[:, 1])
    gf_shapelets = gf_shapelets.reshape((npix, npix))

    fig = plt.figure()
    axis1 = fig.add_subplot(221)
    axis1.set_title("Test Data")
    im1 = axis1.imshow(np.abs(gf_shapelets))
    axis2=fig.add_subplot(222)
    im2 = axis2.imshow(np.abs(codex_shapelets))
    axis2.set_title("Codex Implementation")
    axis3 = fig.add_subplot(212)
    im3 = axis3.imshow(np.abs(codex_shapelets - gf_shapelets))
    axis3.set_title("Difference")

    print("should be writing via matplotlib by now")
    divider = make_axes_locatable(axis1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im1, cax=cax, orientation='vertical')
    divider = make_axes_locatable(axis2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im2, cax=cax, orientation='vertical')
    divider = make_axes_locatable(axis3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im3, cax=cax, orientation='vertical')
    #   plt.show()
    fig.savefig("abs_vals.png")
    plt.close()
    assert np.allclose(np.abs(gf_shapelets), np.abs(codex_shapelets))

def _test_dask_shapelets():
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

        print(np_coords.shape)
        print(np_frequency.shape)
        print(np_coeffs.shape)
        print(np_beta.shape)
        print(np_delta_lm.shape)
        np_shapelets = nb_shapelet(np_coords,
                                np_frequency,
                                np_coeffs,
                                np_beta,
                                np_delta_lm)
        da_shapelets = da_shapelet(da_coords,da_frequency, da_coeffs, da_beta, delta_lm).compute()
        assert_array_almost_equal(da_shapelets, np_shapelets)

def _test_single_shapelet():
    npix = 35
    nrow = npix **2
    nsrc = 1
    nmax = [1, 1]
    beta_vals = [1., 1.]

    u_range = [-3 * np.sqrt(2) *(beta_vals[0] ** (-1)), 3 * np.sqrt(2) * (beta_vals[0] ** (-1))]
    v_range = [-3 * np.sqrt(2) *(beta_vals[1] ** (-1)), 3 * np.sqrt(2) * (beta_vals[1] ** (-1))]

    du = (u_range[1] - u_range[0]) / npix
    dv = (v_range[1] - v_range[0]) / npix
    freqs_u = Fs(np.fft.fftfreq(npix, d=du))
    freqs_v = Fs(np.fft.fftfreq(npix, d=dv))
    uu, vv = np.meshgrid(freqs_u, freqs_v)
    uv = np.vstack((uu.flatten(), vv.flatten())).T


    coords = np.empty((nrow, 3), dtype=np.float)
    coeffs = np.empty((nsrc, nmax[0], nmax[1]), dtype=np.float)
    beta = np.empty((nsrc, 2), dtype=np.float)

    coords[:, :2], coords[:, 2] = uv, 0
    coeffs[0, :, :] = 1
    #coeffs[0, :, :] = 4.
    beta[0, 0], beta[0, 1] = beta_vals[0], beta_vals[1]

    codex_shapelets = nb_shapelet(coords, coeffs, beta).reshape((npix, npix))

    gf_shapelets = np.zeros((nrow), dtype=np.complex128)
    for n1 in range(nmax[0]):
        for n2 in range(nmax[1]):
            gf_shapelets += coeffs[0, n1, n2] * sl.shapelet.computeBasis2d(sl.shapelet.dimBasis2d(n1, n2, beta=beta_vals, fourier=True), uv[:, 0], uv[:, 1])
    gf_shapelets = gf_shapelets.reshape((npix, npix))

    fig = plt.figure()
    axis1 = fig.add_subplot(221)
    axis1.set_title("Test Data")
    im1 = axis1.imshow(np.abs(gf_shapelets))
    axis2=fig.add_subplot(222)
    im2 = axis2.imshow(np.abs(codex_shapelets))
    axis2.set_title("Codex Implementation")
    axis3 = fig.add_subplot(212)
    im3 = axis3.imshow(np.abs(codex_shapelets - gf_shapelets))
    axis3.set_title("Difference")

    print("should be writing via matplotlib by now")
    divider = make_axes_locatable(axis1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im1, cax=cax, orientation='vertical')
    divider = make_axes_locatable(axis2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im2, cax=cax, orientation='vertical')
    divider = make_axes_locatable(axis3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im3, cax=cax, orientation='vertical')
    #   plt.show()
    fig.savefig("abs_vals_single_shapelet.png")
    plt.close()
    assert np.allclose(np.abs(gf_shapelets), np.abs(codex_shapelets))
