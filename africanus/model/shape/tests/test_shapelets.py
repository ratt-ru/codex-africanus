import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
import shapelets as sl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from africanus.model.shape import shapelet as nb_shapelet

Fs = np.fft.fftshift

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

def test_single_shapelet():
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