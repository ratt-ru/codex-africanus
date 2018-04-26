import numpy as np
import pytest

def test_zernike_dde():
    from africanus.rime import zernike_dde
    nsrc = 10
    ntime = 10
    na = 4
    nchan = 10
    n = 5
    m = 4


    lm = np.random.random(size=(nsrc,2))
    frequency = np.linspace(.856e9, .856e8*2, nchan)

    coords = np.empty((3, nsrc, ntime, na, nchan), dtype=np.float)

    for c in range(nchan):
        for a in range(na):
            for t in range(ntime):
                for s in range(nsrc):
                    l_coord, m_coord = lm[s]
                    for c in range(nchan):
                        coords[0, s, t, a, c] = l_coord
                        coords[1, s, t, a, c] = m_coord
                        coords[2, s, t, a, c] = frequency[c]
    zernike_coeffs = zernike_dde(coords, m, n)
    assert zernike_coeffs.shape == (nsrc, ntime, na, nchan)

test_zernike_dde()