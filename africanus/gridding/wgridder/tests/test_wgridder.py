# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal
# import dask
# import dask.array as da
from africanus.constants import c as lightspeed


def explicit_gridder(uvw, freq, ms, wgt, nxdirty, nydirty, xpixsize, ypixsize,
                     apply_w):
    x, y = np.meshgrid(*[-ss/2 + np.arange(ss) for ss in [nxdirty, nydirty]],
                       indexing='ij')
    x *= xpixsize
    y *= ypixsize
    res = np.zeros((nxdirty, nydirty))
    eps = x**2+y**2
    if apply_w:
        nm1 = -eps/(np.sqrt(1.-eps)+1.)
        n = nm1+1
    else:
        nm1 = 0.
        n = 1.
    for row in range(ms.shape[0]):
        for chan in range(ms.shape[1]):
            phase = (freq[chan]/lightspeed *
                     (x*uvw[row, 0] + y*uvw[row, 1] - uvw[row, 2]*nm1))
            if wgt is None:
                res += (ms[row, chan]*np.exp(2j*np.pi*phase)).real
            else:
                res += (ms[row, chan]*wgt[row, chan]
                        * np.exp(2j*np.pi*phase)).real
    return res/n


def test_gridder_mfs():
    # run comparison against explicit gridder where all channels
    # are collapsed into a single MFS image
    from africanus.gridding.wgridder import vis2im
    np.random.seed(420)
    nrow = 1000
    nchan = 8
    fov = 5.0  # degrees
    nx = 32
    ny = 32
    cellx = fov*np.pi/180/nx
    celly = fov*np.pi/180/ny
    f0 = 1e9
    freq = f0 + np.arange(nchan)*(f0/nchan)
    uvw = (np.random.rand(nrow, 3)-0.5)/(cellx*freq[-1]/lightspeed)
    vis = np.random.rand(nrow, nchan)-0.5 + 1j * \
        (np.random.rand(nrow, nchan)-0.5)
    wgt = np.random.rand(nrow, nchan)
    epsilon = 1e-7
    freq_bin_idx = np.arange(1)
    freq_bin_counts = np.array((nchan,), dtype=np.int)
    dirty = vis2im(uvw, freq, vis, wgt, freq_bin_idx, freq_bin_counts,
                   nx, ny, cellx, celly, 2*nx, 2*ny, epsilon, 8, 1).squeeze()

    ref = explicit_gridder(uvw, freq, vis, wgt, nx, ny, cellx, celly, 1)

    # should be within an order of magnitude of precision
    assert_array_almost_equal(dirty, ref, decimal=-int(np.log10(epsilon)) - 1)


def test_gridder():
    # run comparison against dft with full channel resolution
    from africanus.gridding.wgridder import vis2im
    np.random.seed(420)
    nrow = 1000
    nchan = 8
    fov = 5.0  # degrees
    nx = 32
    ny = 32
    cellx = fov*np.pi/180/nx
    celly = fov*np.pi/180/ny
    f0 = 1e9
    freq = f0 + np.arange(nchan)*(f0/nchan)
    uvw = (np.random.rand(nrow, 3)-0.5)/(cellx*freq[-1]/lightspeed)
    vis = np.random.rand(nrow, nchan)-0.5 + 1j * \
        (np.random.rand(nrow, nchan)-0.5)
    wgt = np.random.rand(nrow, nchan)
    epsilon = 1e-7
    freq_bin_idx = np.arange(nchan)
    freq_bin_counts = np.ones(nchan, dtype=np.int)
    dirty = vis2im(uvw, freq, vis, wgt, freq_bin_idx, freq_bin_counts,
                   nx, ny, cellx, celly, 2*nx, 2*ny, epsilon, 8, 1)
    ref = np.zeros((nchan, nx, ny), dtype=np.float64)
    for i in range(nchan):
        ref[i] = explicit_gridder(uvw, freq[i:i+1], vis[:, i:i+1],
                                  wgt[:, i:i+1], nx, ny, cellx, celly, 1)

    # should be within an order of magnitude of precision
    assert_array_almost_equal(dirty, ref, decimal=-int(np.log10(epsilon)) - 1)


def test_freq_mapped_gridder():
    # run comparison against dft with a frequency mapping imposed
    from africanus.gridding.wgridder import vis2im
    np.random.seed(420)
    nrow = 1000
    nchan = 8
    fov = 5.0  # degrees
    nx = 32
    ny = 32
    cellx = fov*np.pi/180/nx
    celly = fov*np.pi/180/ny
    f0 = 1e9
    freq = f0 + np.arange(nchan)*(f0/nchan)
    uvw = (np.random.rand(nrow, 3)-0.5)/(cellx*freq[-1]/lightspeed)
    vis = np.random.rand(nrow, nchan)-0.5 + 1j * \
        (np.random.rand(nrow, nchan)-0.5)
    wgt = np.random.rand(nrow, nchan)
    epsilon = 1e-7
    freq_chunks = 3
    freq_bin_idx = np.arange(0, nchan, freq_chunks)
    freq_mapping = np.append(freq_bin_idx, nchan)
    freq_bin_counts = freq_mapping[1::] - freq_mapping[0:-1]
    dirty = vis2im(uvw, freq, vis, wgt, freq_bin_idx, freq_bin_counts,
                   nx, ny, cellx, celly, 2*nx, 2*ny, epsilon, 8, 1)
    nband = freq_bin_idx.size
    ref = np.zeros((nband, nx, ny), dtype=np.float64)
    for i in range(nband):
        ind = slice(freq_bin_idx[i], freq_bin_idx[i] + freq_bin_counts[i])
        ref[i] = explicit_gridder(uvw, freq[ind], vis[:, ind], wgt[:, ind],
                                  nx, ny, cellx, celly, 1)

    # should be within an order of magnitude of precision
    assert_array_almost_equal(dirty, ref, decimal=-int(np.log10(epsilon)) - 1)


def test_adjointness():
    # instead of explicitly testing the degridder we can just check that
    # it is consistent with the gridder i.e.
    #
    #  <R.H y, x> = <y.H, Rx>
    #
    # where R.H is the gridder, R is the degridder and x and y are randomly
    # drawn image and visibilities respectively
    from africanus.gridding.wgridder import vis2im, im2vis
    np.random.seed(420)
    nrow = 1000
    nchan = 8
    fov = 5.0  # degrees
    nx = 32
    ny = 32
    cellx = fov*np.pi/180/nx
    celly = fov*np.pi/180/ny
    f0 = 1e9
    freq = f0 + np.arange(nchan)*(f0/nchan)
    uvw = (np.random.rand(nrow, 3)-0.5)/(cellx*freq[-1]/lightspeed)
    vis = np.random.rand(nrow, nchan)-0.5 + 1j * \
        (np.random.rand(nrow, nchan)-0.5)
    wgt = np.random.rand(nrow, nchan)
    epsilon = 1e-7
    freq_chunks = 3
    freq_bin_idx = np.arange(0, nchan, freq_chunks)
    freq_mapping = np.append(freq_bin_idx, nchan)
    freq_bin_counts = freq_mapping[1::] - freq_mapping[0:-1]
    dirty = vis2im(uvw, freq, vis, wgt, freq_bin_idx, freq_bin_counts,
                   nx, ny, cellx, celly, 2*nx, 2*ny, epsilon, 8, 1)
    nband = freq_bin_idx.size
    model = np.random.randn(nband, nx, ny)
    modelvis = im2vis(uvw, freq, model, wgt, freq_bin_idx, freq_bin_counts,
                      cellx, celly, 2*nx, 2*ny, epsilon, 8, 1, np.complex128)

    assert_allclose(np.vdot(vis, modelvis).real, np.vdot(dirty, model),
                    rtol=5e-13)


def test_im2residim():
    # Compare the result of im2residim to
    #   VR = V - Rx   - computed with im2vis
    #   IR = R.H VR   - computed with vis2im
    from africanus.gridding.wgridder import vis2im, im2vis, im2residim
    np.random.seed(420)
    nrow = 1000
    nchan = 8
    fov = 5.0  # degrees
    nx = 32
    ny = 32
    cellx = fov*np.pi/180/nx
    celly = fov*np.pi/180/ny
    f0 = 1e9
    freq = f0 + np.arange(nchan)*(f0/nchan)
    uvw = (np.random.rand(nrow, 3)-0.5)/(cellx*freq[-1]/lightspeed)
    vis = np.random.rand(nrow, nchan)-0.5 + 1j * \
        (np.random.rand(nrow, nchan)-0.5)
    wgt = np.random.rand(nrow, nchan)
    epsilon = 1e-7
    freq_chunks = 3
    freq_bin_idx = np.arange(0, nchan, freq_chunks)
    freq_mapping = np.append(freq_bin_idx, nchan)
    freq_bin_counts = freq_mapping[1::] - freq_mapping[0:-1]
    nband = freq_bin_idx.size
    model = np.random.randn(nband, nx, ny)
    modelvis = im2vis(uvw, freq, model, None, freq_bin_idx, freq_bin_counts,
                      cellx, celly, 2*nx, 2*ny, epsilon, 8, 1, np.complex128)
    residualvis = vis - modelvis
    residim1 = vis2im(uvw, freq, residualvis, wgt, freq_bin_idx,
                      freq_bin_counts, nx, ny, cellx, celly, 2*nx, 2*ny,
                      epsilon, 8, 1)

    residim2 = im2residim(uvw, freq, model, vis, wgt, freq_bin_idx,
                          freq_bin_counts, cellx, celly, 2*nx, 2*ny,
                          epsilon, 8, 1)

    assert_array_almost_equal(
        residim1, residim2, decimal=-int(np.log10(epsilon)) - 1)

# if __name__=="__main__":
#     # test_gridder_mfs()
#     # test_gridder()
#     # test_freq_mapped_gridder()
#     # test_adjointness()
#     test_im2residim()
