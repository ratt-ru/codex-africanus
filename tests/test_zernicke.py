import numpy as np
import pytest
import matplotlib.pyplot as plt
from astropy.io import fits
import time

def test_zernike_dde():
    from africanus.rime import zernike_dde
    nsrc = 10
    ntime = 10
    na = 4
    nchan = 10
    npoly=4


    lm = np.random.random(size=(nsrc,2))
    frequency = np.linspace(.856e9, .856e8*2, nchan)

    coords = np.empty((3, nsrc, ntime, na, nchan), dtype=np.float)
    coeffs = np.empty((na, nchan, npoly))
    noll_indices = np.empty((na, nchan, npoly))

    for c in range(nchan):
        for a in range(na):
            coeffs[a,c] = [1,2,3,4]
            noll_indices[a,c] = [5,3,4,2]
            for t in range(ntime):
                for s in range(nsrc):
                    l_coord, m_coord = lm[s]
                    for c in range(nchan):
                        coords[0, s, t, a, c] = l_coord
                        coords[1, s, t, a, c] = m_coord
                        coords[2, s, t, a, c] = frequency[c]
    zernike_value = zernike_dde(coords, coeffs, noll_indices)
    assert zernike_value.shape == (nsrc, ntime, na, nchan)

def _convert_index(j):
    j += 1
    n = 0
    j1 = j-1
    while (j1 > n):
        n += 1
        j1 -= n
    m = (-1)**j * ((n % 2) + 2 * int((j1+((n+1)%2)) / 2.0 ))
    return (n, m)

def write_fits_single(beam, freqs, diameter, filename):
    data = np.zeros((2,)+beam.shape)
    data[0,...] = beam.real
    data[1,...] = beam.imag
    
    # Create header
    hdr = fits.Header()
    fMHz = np.array(freqs)*1e6
    diam = float(diameter)
    ctypes = ['X', 'Y', 'FEED1', 'FEED2', 'PART']
    crvals = [0.0, 0.0, 0, 0, 0]
    cdelts = [diam/beam.shape[-2], diam/beam.shape[-1], 1, 1, 1]
    cunits = ['deg', 'deg', '', '', '']
    nx, ny = beam.shape[-2], beam.shape[-1]
    if nx%2 == 0: crpixx, crpixy = nx/2+0.5, ny/2+0.5
    elif nx%2 == 1: crpixx, crpixy = nx/2+1, ny/2+1
    crpixs = [crpixx, crpixy, 1, 1, 1]
    for i in range(len(data.shape)):
        ii = str(i+1)
        hdr['CTYPE'+ii] = ctypes[i]
        hdr['CRPIX'+ii] = crpixs[i]
        hdr['CRVAL'+ii] = crvals[i]
        hdr['CDELT'+ii] = cdelts[i]
        hdr['CUNIT'+ii] = cunits[i]
    hdr['TELESCOP'] = 'MeerKAT'
    hdr['DATE'] = time.ctime()
    
    # Write real and imag parts of data
    hdu = fits.PrimaryHDU(data, header=hdr)
    hdu.writeto(filename+'.fits', overwrite=True)
"""

def test_zernike_func():
    from africanus.rime import zernike_dde
    Ll = np.deg2rad(179)
    Lm = np.deg2rad(179)
    j = 3
    npix = 256
    nsrc = npix ** 2
    ntime = 1
    na = 1
    nchan = 1
    file_data = np.load('data/meerkat_coeff_dict.npy', encoding='bytes').item()
    eidos_xx_corr = np.load('data/eidos_xx_corr.npy', encoding= 'bytes')
    index_data = file_data.get(b'basei').get(b'xx')
    coeff_data = file_data.get(b'xx')[0]
    nchan = file_data.get(b'xx').shape[0]
    npoly = index_data.shape[0] * 4
    thresh = [15,8]



    # Linear (l,m) grid
    l = np.linspace(-Ll, Ll, npix)
    m = np.linspace(-Lm, Lm, npix)

    ll, mm = np.meshgrid(l,m)
    lm = np.vstack((ll.flatten(),mm.flatten())).T
    
    # Initializing coords, coeffs, and noll_indices
    # coords has nsrc elements (the (l,m) grid), coeffs holds the value 1, and noll_indices has a single value j 
    coords = np.empty((3, nsrc, ntime, na, nchan), dtype=np.float)
    coeffs = np.empty((na, nchan, npoly), dtype=np.complex128)
    noll_indices = np.empty((na, nchan, npoly))
    corrs = [[b'xx', b'xy'], [b'yx', b'yy']]
    for i in range(2):
        for k in range(2):
            thresh_val = thresh[0] if i == j else thresh[1]
            coeff_data = file_data.get(corrs[i][k])[0:thresh_val]
            start_point = i * 2 * coeff_data.shape[0] + k * coeff_data.shape[0]
            end_point = start_point + coeff_data.shape[1]
            coeffs[0,0:thresh_val, start_point:end_point] = coeff_data

            index_data = file_data.get(b'basei').get(corrs[i][k])
            noll_indices[0,0,start_point:end_point] = index_data

    print(lm.shape)
    print(npoly)

    for c in range(nchan):
        coords[0, 0:nsrc, 0, 0, c] = lm[0:nsrc, 0]
        coords[1, 0:nsrc, 0, 0, c] = lm[0:nsrc, 1]
        coords[2, 0:nsrc, 0, 0, c] = 0
        print("looping")
    
    print("Gonna call zernike_dde().")
    zernike_vals = zernike_dde(coords, coeffs, noll_indices)[:, 0, 0, :]
    for n in range(nchan):
        
        zernike_vals[:,0] = np.add(zernike_vals[:, 0], zernike_vals[:, n])
    zernike_vals = zernike_vals[:, 0].reshape((npix, npix))
    print(np.max(np.abs(zernike_vals)))
    zernike_vals_norm = np.divide(zernike_vals , np.max(np.abs(zernike_vals)))
    write_fits_single(zernike_vals, 1, 1, './zernicke_test_figs/eidos_primary_beam' )
    n, m = _convert_index(j)
    plt.figure()
    plt.title(r'$Z_{%d}^{%d}(l,m)$' %(n, m))
    plt.imshow(np.real(zernike_vals_norm))
    plt.colorbar()
    plt.savefig('./zernicke_test_figs/nsrc_%d_ntime_%d_npoly_%d_na_%d_nchan_%d.png' %(nsrc, ntime, npoly, na, nchan))

"""

def test_zernike_func_xx_corr():
    from africanus.rime import zernike_dde
    j = 3
    npix = 257
    nsrc = npix ** 2
    ntime = 1
    na = 1
    nchan = 1
    file_data = np.load('data/meerkat_coeff_dict.npy', encoding='bytes').item()
    eidos_corr_vals = np.load('data/eidos_xx_corr.npy', encoding= 'bytes')
    nchan = 1
    thresh_vals = [15,8]
    corrs = [[b'xx', b'xy'], [b'yx', b'yy']]
    for i in range(2):
        for j in range(2):
            #Get the correct correlation, and set appropriate value for npoly
            thresh = 0
            if(i == j):
                thresh = thresh_vals[0]
            else:
                thresh = thresh_vals[1]
            npoly = thresh

            # Linear (l,m) grid
            nx, ny = npix, npix
            grid = (np.indices((nx, ny), dtype=np.float) - nx//2) * 2 / nx
            ll, mm =  grid[0], grid[1]

            lm = np.vstack((ll.flatten(),mm.flatten())).T

            # Initializing coords, coeffs, and noll_indices 
            coords = np.empty((3, nsrc, ntime, na, nchan), dtype=np.float)
            coeffs = np.empty((na, nchan, npoly), dtype=np.complex128)
            noll_indices = np.empty((na, nchan, npoly))

            #Assign Values to coeffs and noll_indices
            coeffs[0,0,:] = file_data.get(corrs[i][j])[644,:thresh]
            noll_indices[0,0,:] = file_data.get(b'basei').get(corrs[i][j])[:thresh]

            # I left 0 as all the freq values
            coords[0, 0:nsrc, 0, 0, 0] = lm[0:nsrc, 0]
            coords[1, 0:nsrc, 0, 0, 0] = lm[0:nsrc, 1]
            coords[2, 0:nsrc, 0, 0, 0] = np.zeros(nsrc)

            # Call the function, reshape accordingly, and normalise
            zernike_vals = zernike_dde(coords, coeffs, noll_indices)[:, 0, 0, 0].reshape((npix, npix))
            zernike_vals_norm = zernike_vals
            eidos_corr = eidos_corr_vals[i,j]

            # Create graphs of Zernicke, Eidos, and the difference between them
            write_fits_single(zernike_vals, 1, 1, './zernicke_test_figs/eidos_primary_beam' )
            plt.figure()
            plt.title('Codex_Zernicke' )
            plt.imshow(np.real(zernike_vals))
            plt.colorbar()
            plt.savefig('./zernicke_test_figs/Codex_nsrc_%d_ntime_%d_npoly_%d_na_%d_nchan_%d_%s.png' %(nsrc, ntime, npoly, na, nchan, corrs[i][j]))
            print("%s Correlation: %r" %(corrs[i][j], np.allclose(np.real(eidos_corr), np.real(zernike_vals))))

            diff = np.real(zernike_vals_norm) - np.real(eidos_corr)

            plt.figure()
            plt.title('Eidos_Zernicke')
            plt.imshow(np.real(eidos_corr))
            plt.colorbar()
            plt.savefig('./zernicke_test_figs/Eidos_nsrc_%d_ntime_%d_npoly_%d_na_%d_nchan_%d_%s.png' %(nsrc, ntime, npoly, na, nchan, corrs[i][j]))

            plt.figure()
            plt.title('Difference')
            plt.imshow(diff)
            plt.colorbar()
            plt.savefig('./zernicke_test_figs/Difference_nsrc_%d_ntime_%d_npoly_%d_na_%d_nchan_%d.png' %(nsrc, ntime, npoly, na, nchan))



    
test_zernike_func_xx_corr()