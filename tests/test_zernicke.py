import numpy as np
import pytest
import matplotlib.pyplot as plt

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

def test_zernike_func():
    from africanus.rime import zernike_dde
    Ll = np.deg2rad(3)
    Lm = np.deg2rad(3)
    j = 1
    nsrc = 10000
    ntime = 1
    npoly = 1
    na = 1
    nchan = 1

    # Linear (l,m) grid
    l = np.linspace(-Ll, Ll, nsrc)
    m = np.linspace(-Lm, Lm, nsrc)
    
    # Initializing coords, coeffs, and noll_indices
    # coords has nsrc elements (the (l,m) grid), coeffs holds the value 1, and noll_indices has a single value j 
    coords = np.empty((3, nsrc, ntime, na, nchan), dtype=np.float)
    coeffs = np.empty((na, nchan, npoly))
    coeffs[0, 0, 0] = 1
    noll_indices = np.empty((na, nchan, npoly))
    noll_indices[0,0,0] = j

    # I left 0 as all the freq values
    for s in range(nsrc):
        coords[0, s, 0, 0, 0] = l[s]
        coords[1, s, 0, 0, 0] = m[s]
        coords[2, s, 0, 0, 0] = 0
    zernike_vals = zernike_dde(coords, coeffs, noll_indices)[:, 0, 0, 0].reshape((int(np.sqrt(nsrc)), int(np.sqrt(nsrc))))

    plt.figure()
    plt.title(r'$Z_{%d}(l,m)$' %j)
    plt.imshow(np.real(zernike_vals))
    plt.colorbar()
    plt.savefig('./zernicke_test_figs/nsrc_10000_ntime_1_npoly_1_na_1_nchan_1.png')


    
test_zernike_func()