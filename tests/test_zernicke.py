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

test_zernike_dde()