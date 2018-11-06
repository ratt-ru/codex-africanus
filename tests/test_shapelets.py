import numpy as np
import pytest

def test_shapelets():
    from africanus.rime import shapelet_dde
    npix = 17
    nsrc = npix ** 2
    ntime = 1
    na = 1
    nchan = 1
    npoly = 3

    grid = (np.indices((npix, npix), dtype=np.float) - npix//2) * 2 / npix
    ll, mm =  grid[0], grid[1]

    lm = np.vstack((ll.flatten(),mm.flatten())).T

    coords = np.empty((3, nsrc, ntime, na, nchan), dtype=np.float)
    assert True
