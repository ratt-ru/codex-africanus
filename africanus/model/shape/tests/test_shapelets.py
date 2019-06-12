import numpy as np
from numpy.testing import assert_array_almost_equal
#import pytest
from shapelets import shapelet as sl
from scipy.special import factorial, hermite
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from africanus.model.shape import shapelet as nb_shapelet

Fs = np.fft.fftshift
iFs = np.fft.ifftshift

def shapelet_img_space(xx, n, beta):
    basis_component = ((2**n) * ((np.pi)**(0.5)) * factorial(n) * beta)**(-0.5)
    exponential_component = hermite(n)(xx / beta) * np.exp((-0.5) * (xx**2) * (beta **(-2)))
    return basis_component * exponential_component

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

        row_chunks = (2,2)
        source_chunks = (5,10,5,5)
        
        row = sum(row_chunks)
        source = sum(source_chunks)
        nmax = [5, 5]
        beta_vals = [1., 1.]

        np_coords = np.random.randn(row, 3)
        np_coeffs = np.random.randn(source, nmax[0], nmax[1])
        np_beta = np.empty((source, 2))
        np_beta[:, 0], np_beta[:, 1] = beta_vals[0], beta_vals[1]

        da_coords = da.from_array(np_coords, chunks=(row_chunks, 3))
        da_coeffs = da.from_array(np_coeffs, chunks=(source_chunks, nmax[0], nmax[1]))
        da_beta = da.from_array(np_beta, chunks=(source_chunks, 2))
        
        np_shapelets = nb_shapelet(np_coords,
                                np_coeffs,
                                np_beta)
        da_shapelets = da_shapelet(da_coords, da_coeffs, da_beta).compute()
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