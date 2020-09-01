# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal
import pytest
from africanus.constants import c as lightspeed

pmp = pytest.mark.parametrize


def _l2error(a, b):
    return np.sqrt(np.sum(np.abs(a-b)**2)/np.maximum(np.sum(np.abs(a)**2),
                                                     np.sum(np.abs(b)**2)))


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


@pmp("nx", (32,))
@pmp("ny", (18, 64))
@pmp("fov", (1.0, 5.0))
@pmp("nrow", (1000,))
@pmp("nchan", (1, 7))
@pmp("nband", (1, 3))
@pmp("epsilon", (1e-5, 1e-10))
@pmp("wstacking", (True, False))
@pmp("nthreads", (1, 6))
def test_gridder(nx, ny, fov, nrow, nchan, nband,
                 epsilon, wstacking, nthreads):
    # run comparison against dft with a frequency mapping imposed
    if nband > nchan:
        return
    from africanus.gridding.wgridder import vis2im
    if epsilon >= 5e-6:
        real_type = "f4"
        complex_type = "c8"
    else:
        real_type = "f8"
        complex_type = "c16"
    np.random.seed(420)
    cellx = fov*np.pi/180/nx
    celly = fov*np.pi/180/ny * 1.1  # to test different cell sizes
    f0 = 1e9
    freq = (f0 + np.arange(nchan)*(f0/nchan))
    uvw = ((np.random.rand(nrow, 3)-0.5) /
           (cellx*freq[-1]/lightspeed))
    vis = (np.random.rand(nrow, nchan)-0.5 + 1j *
           (np.random.rand(nrow, nchan)-0.5)).astype(complex_type)
    wgt = np.random.rand(nrow, nchan).astype(real_type)
    step = nchan//nband
    if step:
        freq_bin_idx = np.arange(0, nchan, step)
        freq_mapping = np.append(freq_bin_idx, nchan)
        freq_bin_counts = freq_mapping[1::] - freq_mapping[0:-1]
    else:
        freq_bin_idx = np.array([0], dtype=np.int16)
        freq_bin_counts = np.array([1], dtype=np.int16)
    dirty = vis2im(uvw, freq, vis, wgt, freq_bin_idx, freq_bin_counts,
                   nx, ny, cellx, celly, 2*nx, 2*ny,
                   epsilon, nthreads, wstacking)
    nband = freq_bin_idx.size
    ref = np.zeros((nband, nx, ny), dtype=np.float64)
    for i in range(nband):
        ind = slice(freq_bin_idx[i], freq_bin_idx[i] + freq_bin_counts[i])
        ref[i] = explicit_gridder(uvw, freq[ind], vis[:, ind], wgt[:, ind],
                                  nx, ny, cellx, celly,
                                  wstacking)

    # l2 error should be within epsilon of zero
    assert_allclose(_l2error(dirty, ref), 0, atol=epsilon)


@pmp("nx", (30,))
@pmp("ny", (50, 128))
@pmp("fov", (0.5, 2.5))
@pmp("nrow", (333, 5000,))
@pmp("nchan", (1, 4))
@pmp("nband", (1, 2))
@pmp("epsilon", (1e-5, 1e-10,))
@pmp("wstacking", (True, False))
@pmp("nthreads", (6,))
def test_adjointness(nx, ny, fov, nrow, nchan, nband,
                     epsilon, wstacking, nthreads):
    # instead of explicitly testing the degridder we can just check that
    # it is consistent with the gridder i.e.
    #
    #  <R.H y, x> = <y.H, Rx>
    #
    # where R.H is the gridder, R is the degridder and x and y are randomly
    # drawn image and visibilities respectively
    if nband > nchan:
        return
    from africanus.gridding.wgridder import vis2im, im2vis
    if epsilon >= 5e-6:
        real_type = "f4"
        complex_type = "c8"
    else:
        real_type = "f8"
        complex_type = "c16"
    np.random.seed(420)
    cellx = fov*np.pi/180/nx
    celly = fov*np.pi/180/ny*0.9  # to test with different cell sizes
    f0 = 1e9
    freq = (f0 + np.arange(nchan)*(f0/nchan))
    uvw = ((np.random.rand(nrow, 3)-0.5) /
           (cellx*freq[-1]/lightspeed))
    vis = (np.random.rand(nrow, nchan)-0.5 + 1j *
           (np.random.rand(nrow, nchan)-0.5)).astype(complex_type)
    wgt = np.random.rand(nrow, nchan).astype(real_type)
    step = nchan//nband
    if step:
        freq_bin_idx = np.arange(0, nchan, step)
        freq_mapping = np.append(freq_bin_idx, nchan)
        freq_bin_counts = freq_mapping[1::] - freq_mapping[0:-1]
    else:
        freq_bin_idx = np.array([0], dtype=np.int8)
        freq_bin_counts = np.array([1], dtype=np.int8)
    nband = freq_bin_idx.size
    dirty = vis2im(uvw, freq, vis, wgt, freq_bin_idx, freq_bin_counts,
                   nx, ny, cellx, celly, 2*nx, 2*ny, epsilon, nthreads,
                   wstacking)
    model = np.random.randn(nband, nx, ny).astype(real_type)
    modelvis = im2vis(uvw, freq, model, wgt, freq_bin_idx, freq_bin_counts,
                      cellx, celly, 2*nx, 2*ny, epsilon, nthreads, wstacking,
                      complex_type)

    # should have relative tolerance close to machine precision
    tol = 5e-5 if epsilon >= 5e-6 else 5e-13
    assert_allclose(np.vdot(vis, modelvis).real, np.vdot(dirty, model),
                    rtol=tol)


@pmp("nx", (20, ))
@pmp("ny", (32, 70))
@pmp("fov", (1.5, 3.5))
@pmp("nrow", (222, 777,))
@pmp("nchan", (1, 5))
@pmp("nband", (1, 3))
@pmp("epsilon", (1e-5, 1e-10))
@pmp("wstacking", (True, False))
@pmp("nthreads", (3,))
def test_im2residim(nx, ny, fov, nrow, nchan, nband,
                    epsilon, wstacking, nthreads):
    # Compare the result of im2residim to
    #   VR = V - Rx   - computed with im2vis
    #   IR = R.H VR   - computed with vis2im
    from africanus.gridding.wgridder import vis2im, im2vis, im2residim
    np.random.seed(420)
    if epsilon >= 5e-6:
        real_type = "f4"
        complex_type = "c8"
    else:
        real_type = "f8"
        complex_type = "c16"
    cellx = fov*np.pi/180/nx
    celly = fov*np.pi/180/ny * 1.1
    f0 = 1e9
    freq = (f0 + np.arange(nchan)*(f0/nchan))
    uvw = ((np.random.rand(nrow, 3)-0.5) /
           (cellx*freq[-1]/lightspeed))
    vis = (np.random.rand(nrow, nchan)-0.5 + 1j *
           (np.random.rand(nrow, nchan)-0.5)).astype(complex_type)
    wgt = np.random.rand(nrow, nchan).astype(real_type)
    step = nchan//nband
    if step:
        freq_bin_idx = np.arange(0, nchan, step)
        freq_mapping = np.append(freq_bin_idx, nchan)
        freq_bin_counts = freq_mapping[1::] - freq_mapping[0:-1]
    else:
        freq_bin_idx = np.array([0], dtype=np.int8)
        freq_bin_counts = np.array([1], dtype=np.int8)
    nband = freq_bin_idx.size
    model = np.random.randn(nband, nx, ny).astype(real_type)
    modelvis = im2vis(uvw, freq, model, None, freq_bin_idx, freq_bin_counts,
                      cellx, celly, 2*nx, 2*ny, epsilon, nthreads, wstacking,
                      complex_type)
    residualvis = vis - modelvis
    residim1 = vis2im(uvw, freq, residualvis, wgt, freq_bin_idx,
                      freq_bin_counts, nx, ny, cellx, celly, 2*nx, 2*ny,
                      epsilon, nthreads, wstacking)

    residim2 = im2residim(uvw, freq, model, vis, wgt, freq_bin_idx,
                          freq_bin_counts, cellx, celly, 2*nx, 2*ny,
                          epsilon, nthreads, wstacking)

    # These are essentially computing the same thing just in a different
    # order so should be close to machine precision
    decimal = 5 if epsilon >= 5e-6 else 10
    rmax = np.maximum(np.abs(residim1).max(), np.abs(residim2).max())
    assert_array_almost_equal(
        residim1/rmax, residim2/rmax, decimal=decimal)


@pmp("nx", (30, 250))
@pmp("ny", (128,))
@pmp("fov", (5.0,))
@pmp("nrow", (3333, 10000))
@pmp("nchan", (1, 8))
@pmp("nband", (1, 2))
@pmp("epsilon", (1e-5, 1e-10))
@pmp("wstacking", (False, True))
@pmp("nthreads", (1, 4))
@pmp("nchunks", (1, 3))
def test_dask_vis2im(nx, ny, fov, nrow, nchan, nband,
                     epsilon, wstacking, nthreads, nchunks):
    da = pytest.importorskip("dask.array")
    from africanus.gridding.wgridder import vis2im as vis2im_np
    from africanus.gridding.wgridder.dask import vis2im
    np.random.seed(420)
    if epsilon >= 5e-6:
        real_type = "f4"
        complex_type = "c8"
    else:
        real_type = "f8"
        complex_type = "c16"
    cellx = fov*np.pi/180/nx
    celly = fov*np.pi/180/ny * 1.1
    f0 = 1e9
    freq = (f0 + np.arange(nchan)*(f0/nchan))
    uvw = ((np.random.rand(nrow, 3)-0.5) /
           (cellx*freq[-1]/lightspeed))
    vis = (np.random.rand(nrow, nchan)-0.5 + 1j *
           (np.random.rand(nrow, nchan)-0.5)).astype(complex_type)
    wgt = np.random.rand(nrow, nchan).astype(real_type)
    step = np.maximum(1, nchan//nband)
    if step:
        freq_bin_idx = np.arange(0, nchan, step)
        freq_mapping = np.append(freq_bin_idx, nchan)
        freq_bin_counts = freq_mapping[1::] - freq_mapping[0:-1]
    else:
        freq_bin_idx = np.array([0], dtype=np.int8)
        freq_bin_counts = np.array([1], dtype=np.int8)
    nband = freq_bin_idx.size
    dirty = vis2im_np(uvw, freq, vis, wgt, freq_bin_idx, freq_bin_counts,
                      nx, ny, cellx, celly, 2*nx, 2*ny, epsilon, nthreads,
                      wstacking)

    # now get result using dask
    rows_per_task = int(np.ceil(nrow/nchunks))
    row_chunks = (nchunks-1) * (rows_per_task,)
    row_chunks += (nrow - np.sum(row_chunks),)
    freq_da = da.from_array(freq, chunks=step)
    uvw_da = da.from_array(uvw, chunks=(row_chunks, -1))
    vis_da = da.from_array(vis, chunks=(row_chunks, step))
    wgt_da = da.from_array(wgt, chunks=(row_chunks, step))
    freq_bin_idx_da = da.from_array(freq_bin_idx, chunks=1)
    freq_bin_counts_da = da.from_array(freq_bin_counts, chunks=1)

    dirty_da = vis2im(uvw_da, freq_da, vis_da, wgt_da, freq_bin_idx_da,
                      freq_bin_counts_da, nx, ny, cellx, celly, 2*nx, 2*ny,
                      epsilon, nthreads, wstacking).compute()

    # relative error should agree to within epsilon
    dmax = np.maximum(np.abs(dirty).max(), np.abs(dirty_da).max())
    assert_array_almost_equal(dirty/dmax, dirty_da/dmax,
                              decimal=-int(np.log10(epsilon)))


@pmp("nx", (34, 28))
@pmp("ny", (128,))
@pmp("fov", (5.0,))
@pmp("nrow", (444, 5050,))
@pmp("nchan", (1, 4))
@pmp("nband", (1, 3))
@pmp("epsilon", (1e-5, 1e-10))
@pmp("wstacking", (False, True))
@pmp("nthreads", (1, 4))
@pmp("nchunks", (1, 3))
def test_dask_im2vis(nx, ny, fov, nrow, nchan, nband,
                     epsilon, wstacking, nthreads, nchunks):
    da = pytest.importorskip("dask.array")
    from africanus.gridding.wgridder import im2vis as im2vis_np
    from africanus.gridding.wgridder.dask import im2vis
    np.random.seed(420)
    if epsilon >= 5e-6:
        real_type = "f4"
        complex_type = "c8"
    else:
        real_type = "f8"
        complex_type = "c16"
    cellx = fov*np.pi/180/nx
    celly = fov*np.pi/180/ny
    f0 = 1e9
    freq = (f0 + np.arange(nchan)*(f0/nchan))
    uvw = ((np.random.rand(nrow, 3)-0.5) /
           (cellx*freq[-1]/lightspeed))
    vis = (np.random.rand(nrow, nchan)-0.5 + 1j *
           (np.random.rand(nrow, nchan)-0.5)).astype(complex_type)
    wgt = np.random.rand(nrow, nchan).astype(real_type)

    step = np.maximum(1, nchan//nband)
    if step:
        freq_bin_idx = np.arange(0, nchan, step)
        freq_mapping = np.append(freq_bin_idx, nchan)
        freq_bin_counts = freq_mapping[1::] - freq_mapping[0:-1]
    else:
        freq_bin_idx = np.array([0], dtype=np.int16)
        freq_bin_counts = np.array([1], dtype=np.int16)
    nband = freq_bin_idx.size
    model = np.random.randn(nband, nx, ny).astype(real_type)

    vis = im2vis_np(uvw, freq, model, wgt, freq_bin_idx, freq_bin_counts,
                    cellx, celly, 2*nx, 2*ny, epsilon, nthreads, wstacking,
                    complex_type)

    # now get result using dask
    rows_per_task = int(np.ceil(nrow/nchunks))
    row_chunks = (nchunks-1) * (rows_per_task,)
    row_chunks += (nrow - np.sum(row_chunks),)
    freq_da = da.from_array(freq, chunks=step)
    uvw_da = da.from_array(uvw, chunks=(row_chunks, -1))
    model_da = da.from_array(model, chunks=(1, nx, ny))
    wgt_da = da.from_array(wgt, chunks=(row_chunks, step))
    freq_bin_idx_da = da.from_array(freq_bin_idx, chunks=1)
    freq_bin_counts_da = da.from_array(freq_bin_counts, chunks=1)

    vis_da = im2vis(uvw_da, freq_da, model_da, wgt_da, freq_bin_idx_da,
                    freq_bin_counts_da, cellx, celly, 2*nx, 2*ny, epsilon,
                    nthreads, wstacking, complex_type).compute()

    # relative error should agree to within epsilon
    vmax = np.maximum(np.abs(vis).max(), np.abs(vis_da).max())
    assert_array_almost_equal(vis/vmax, vis_da/vmax,
                              decimal=-int(np.log10(epsilon)))


@pmp("nx", (54, 62))
@pmp("ny", (28,))
@pmp("fov", (5.0,))
@pmp("nrow", (1111,))
@pmp("nchan", (1, 6))
@pmp("nband", (1, 4))
@pmp("epsilon", (1e-5, 1e-10))
@pmp("wstacking", (False, True))
@pmp("nthreads", (1, 4))
@pmp("nchunks", (1, 3))
def test_dask_im2residim(nx, ny, fov, nrow, nchan, nband,
                         epsilon, wstacking, nthreads, nchunks):
    da = pytest.importorskip("dask.array")
    from africanus.gridding.wgridder import im2residim as im2residim_np
    from africanus.gridding.wgridder.dask import im2residim
    np.random.seed(420)
    if epsilon >= 5e-6:
        real_type = "f4"
        complex_type = "c8"
    else:
        real_type = "f8"
        complex_type = "c16"
    cellx = fov*np.pi/180/nx
    celly = fov*np.pi/180/ny * 1.1
    f0 = 1e9
    freq = (f0 + np.arange(nchan)*(f0/nchan))
    uvw = ((np.random.rand(nrow, 3)-0.5) /
           (cellx*freq[-1]/lightspeed))
    vis = (np.random.rand(nrow, nchan)-0.5 + 1j *
           (np.random.rand(nrow, nchan)-0.5)).astype(complex_type)
    wgt = np.random.rand(nrow, nchan).astype(real_type)
    step = np.maximum(1, nchan//nband)
    if step:
        freq_bin_idx = np.arange(0, nchan, step)
        freq_mapping = np.append(freq_bin_idx, nchan)
        freq_bin_counts = freq_mapping[1::] - freq_mapping[0:-1]
    else:
        freq_bin_idx = np.array([0], dtype=np.int8)
        freq_bin_counts = np.array([1], dtype=np.int8)
    nband = freq_bin_idx.size
    model = np.random.randn(nband, nx, ny).astype(real_type)
    residim_np = im2residim_np(uvw, freq, model, vis, wgt, freq_bin_idx,
                               freq_bin_counts, cellx, celly, 2*nx, 2*ny,
                               epsilon, nthreads, wstacking)

    rows_per_task = int(np.ceil(nrow/nchunks))
    row_chunks = (nchunks-1) * (rows_per_task,)
    row_chunks += (nrow - np.sum(row_chunks),)
    freq_da = da.from_array(freq, chunks=step)
    uvw_da = da.from_array(uvw, chunks=(row_chunks, -1))
    model_da = da.from_array(model, chunks=(1, nx, ny))
    vis_da = da.from_array(vis, chunks=(row_chunks, step))
    wgt_da = da.from_array(wgt, chunks=(row_chunks, step))
    freq_bin_idx_da = da.from_array(freq_bin_idx, chunks=1)
    freq_bin_counts_da = da.from_array(freq_bin_counts, chunks=1)

    residim_da = im2residim(uvw_da, freq_da, model_da, vis_da, wgt_da,
                            freq_bin_idx_da, freq_bin_counts_da,
                            cellx, celly, 2*nx, 2*ny,
                            epsilon, nthreads, wstacking).compute()

    # should agree to within epsilon
    rmax = np.maximum(np.abs(residim_np).max(), np.abs(residim_da).max())
    assert_array_almost_equal(
        residim_np/rmax, residim_da/rmax, decimal=-int(np.log10(epsilon)))
