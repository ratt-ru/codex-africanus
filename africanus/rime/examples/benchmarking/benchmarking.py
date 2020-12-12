import numpy as np
from africanus.rime.dask import zernike_dde
import dask.array as da
from dask.distributed import profile, Client
import dask
from multiprocessing.pool import ThreadPool
import time
import matplotlib.pyplot as plt
import argparse
import os
import packratt
from zernike_helper_funcs import _convert_coords, zernike_func, noll_to_zern

codex_grid = None

def read_coeffs(filename, frequency, na, npoly=20):
    coeffs_r = np.empty((na, len(frequency), 2,2,npoly))
    coeffs_i = np.empty((na, len(frequency), 2,2,npoly))
    noll_index_r = np.empty((na, len(frequency), 2,2,npoly))
    noll_index_i = np.empty((na, len(frequency), 2,2,npoly))
    if not os.path.exists("./meerkat/"): packratt.get(filename, "./")
    c_freqs = np.load("./meerkat/freqs.npy", allow_pickle=True)
    ch = [abs(c_freqs - i/1e06).argmin() for i in (frequency)]
    params = np.load("./meerkat/meerkat_zernike_coeffs.npz/params.npy", allow_pickle=True)
    for ant in range(na):
        for chan in range(len(frequency)):
            coeffs_r[ant, chan, :,:,:] = params[chan,0][0,:,:,:]
            coeffs_i[ant, chan, :,:,:] = params[chan,0][1,:,:,:]
            noll_index_r[ant, chan, :,:,:] = params[chan,1][0,:,:,:]
            noll_index_i[ant, chan, :,:,:] = params[chan,1][1,:,:,:]

    return ((coeffs_r, noll_index_r),(coeffs_i, noll_index_i))



def codex_unit_disk(ntime, na, nchan, npix=512):
    if debugCode:
        npix = 32
    coords = np.empty((3, npix**2, ntime, na, nchan))
    coords[2,...] = 0
    nx, ny = npix, npix
    grid = (np.indices((nx, ny), dtype=np.float) - nx/2) / (nx*1./2)
    grid0 = grid[0,:,:].flatten()
    grid1 = grid[1,:,:].flatten()
    # quit()
    for time in range(ntime):
        for ant in range(na):
            for freq in range(nchan):
                coords[0,:,time, ant, freq] = grid0
                coords[1,:,time,ant,freq] = grid1
                

    return coords

def create_zernike_beam(cores, npix):
    global codex_grid
    client = Client()
    dask.config.set(pool=ThreadPool(cores))
    pix_chunks = (npix**2) // cores
    ((coeffs_r, noll_index_r),(coeffs_i, noll_index_i)) = read_coeffs("/beams/meerkat/meerkat_zernike_coeffs/meerkat/zernike_coeffs.tar.gz", frequency, na)

    parallactic_angles = np.zeros((ntime, na))
    pointing_errors = np.zeros((ntime, na, nchan, 2))
    antenna_scaling = np.ones((na, nchan, 2))

    coord_chunks = (3, pix_chunks, 1, 1, nchan)
    coeff_chunks = (1, nchan, 2, 2, 20)
    as_chunks = (1, nchan, 2)
    pa_chunks = (1, 1)
    pe_chunks = (1, 1, nchan, 2)

    # print(coord_chunks)
    t_grid_0 = time.time()
    np_coords = codex_unit_disk(ntime, na, nchan, npix=npix)
    codex_grid = np_coords
    t_grid_1 = time.time()
    print("Gridding time: ", t_grid_1 - t_grid_1, " seconds")
    # plt.figure()
    # plt.plot(np_coords[0,:,0,0,0])
    # plt.plot(np_coords[1,:,0,0,0])
    # plt.show()
    # quit()
    da_coords = da.from_array(np_coords, chunks=coord_chunks)
    da_freq = da.from_array(np.ones(nchan))
    da_pa = da.from_array(parallactic_angles, chunks=pa_chunks)
    da_pe = da.from_array(pointing_errors, chunks=pe_chunks)
    da_as = da.from_array(antenna_scaling, chunks=as_chunks)
    da_coeffs_r = da.from_array(coeffs_r, chunks=coeff_chunks)
    da_coeffs_i = da.from_array(coeffs_i, chunks=coeff_chunks)
    da_ni_r = da.from_array(noll_index_r, chunks=coeff_chunks)
    da_ni_i = da.from_array(noll_index_i, chunks=coeff_chunks)

    # print(np_coords)
    # quit()
    t0 = time.time()
    zernike_beam = (zernike_dde(da_coords, da_coeffs_r, da_ni_r, da_pa, da_freq, da_as, da_pe) +
                1j * zernike_dde(da_coords, da_coeffs_i, da_ni_i, da_pa, da_freq, da_as, da_pe)).compute()
    # client.compute(zernike_beam)
    # client.profile(filename="dask_profile.html", plot_data=True)
    
    t1 = time.time()
    return zernike_beam



def numpy_zernike(coords, coeffs, noll_index, parallactic_angles, frequency_scaling, antenna_scaling, pointing_errors):
    l_coords, m_coords = coords[0,...], coords[1,...]
    rho, phi = np.empty(l_coords.shape), np.empty(m_coords.shape)
    for a in range(l_coords.shape[0]):
        for b in range(l_coords.shape[1]):
            for c in range(l_coords.shape[2]):
                for d in range(l_coords.shape[3]):
                    rho[a,b,c,d], phi[a,b,c,d] = _convert_coords(l_coords[a,b,c,d], m_coords[a,b,c,d])
    # (rho, phi) = _convert_coords(coords[0,:,:,:,:], coords[1,:,:,:,:])
    a,b,c,d = np.where(rho > 1)
    n_indices, m_indices = noll_to_zern(noll_index)
    zern = None
    for i in range(20):
        n, m = n_indices[:,:,0,0,i], m_indices[:,:,0,0,i]
        c = coeffs[0,:,0,0,i]
        if zern is None:
            zern = np.einsum("c,rtac->rtac",c, zernike_func(n, m, rho, phi))
        else:
            zern = zern + np.einsum("c,rtac->rtac",c, zernike_func(n, m, rho, phi))
    return zern
    

def numpy_implementation(npix):
    coords = codex_unit_disk(ntime, na, nchan, npix=npix)
    parallactic_angles = np.zeros((ntime, na))
    pointing_errors = np.zeros((ntime, na, nchan, 2))
    antenna_scaling = np.ones((na, nchan, 2))
    frequency_scaling = np.ones(nchan)

    ((coeffs_r, noll_index_r),(coeffs_i, noll_index_i)) = read_coeffs("/beams/meerkat/meerkat_zernike_coeffs/meerkat/zernike_coeffs.tar.gz", frequency, na)

    zern_real = numpy_zernike(coords, coeffs_r, noll_index_r, parallactic_angles, frequency_scaling, antenna_scaling, pointing_errors)
    zern_imag = numpy_zernike(coords, coeffs_i, noll_index_i, parallactic_angles, frequency_scaling, antenna_scaling, pointing_errors)

    return zern_real + (1j * zern_imag)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-c", "--cores", type=int, default=8)
    args.add_argument("-n", "--num-pixels", type=int, default=512)
    args = args.parse_args()
    cores = args.cores
    debugCode = False
    t0 = time.time()
    ntime = 1
    na = 1
    frequency = [1080000000]
    nchan = len(frequency)

    npix = args.num_pixels

    # Time Numba implementation
    t_numba_0 = time.time()
    zernike_beam = create_zernike_beam(cores, npix)
    t_numba_1 = time.time()

    print("Numba implementation: ", t_numba_1 - t_numba_0)
    print("Numba maximum: ", np.max(zernike_beam))
    # quit()
    # Time NumPy implementation
    t_numpy_0 = time.time()
    numpy_zernike_beam = numpy_implementation(npix)
    t_numpy_1 = time.time()
    print("NumPy implementation: ", t_numpy_1 - t_numpy_0)
    print("NumPy maximum: ", np.max(numpy_zernike_beam))

    print("Beams match? ", np.allclose(zernike_beam[...,0,0], numpy_zernike_beam))

    plt.figure()
    plt.imshow(zernike_beam.real[:,0,0,0,0,0].reshape((npix,npix)))
    plt.title("Numba Beam")
    plt.colorbar()
    plt.show()
    plt.close()

    plt.figure()
    plt.imshow(numpy_zernike_beam.real[:,0,0,0].reshape((npix,npix)))
    plt.title("Numba Beam")
    plt.colorbar()
    plt.show()

    # print("Zernike Code: ", t1 - t0, " seconds")
    # print(np.sqrt(zernike_beam.shape[0]))
    # print(zernike_beam.shape)

    # plt.figure()
    # plt.imshow(zernike_beam.real[:,0,0,0,0,0].reshape(npix,npix))
    # plt.title("Real")
    # # plt.show()
    # # plt.close()


    # plt.figure()
    # plt.imshow(zernike_beam.imag[:,0,0,0,0,0].reshape(npix,npix))
    # plt.title("Imaginary")
    # plt.show()
    # plt.close()