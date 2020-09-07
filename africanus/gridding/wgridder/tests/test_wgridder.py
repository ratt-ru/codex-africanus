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


@pmp("nx", (16,))
@pmp("ny", (18, 64))
@pmp("fov", (5.0,))
@pmp("nrow", (1000,))
@pmp("nchan", (1, 7))
@pmp("nband", (1, 3))
@pmp("precision", ('single', 'double'))
@pmp("nthreads", (1, 6))
def test_gridder(nx, ny, fov, nrow, nchan, nband,
                 precision, nthreads):
    # run comparison against dft with a frequency mapping imposed
    if nband > nchan:
        return
    from africanus.gridding.wgridder import dirty
    if precision == 'single':
        real_type = "f4"
        complex_type = "c8"
        epsilon = 1e-4
    else:
        real_type = "f8"
        complex_type = "c16"
        epsilon = 1e-7
    np.random.seed(420)
    cell = fov*np.pi/180/nx
    f0 = 1e9
    freq = (f0 + np.arange(nchan)*(f0/nchan))
    uvw = ((np.random.rand(nrow, 3)-0.5) /
           (cell*freq[-1]/lightspeed))
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
    image = dirty(uvw, freq, vis, freq_bin_idx, freq_bin_counts, nx, ny, cell,
                  weights=wgt, nthreads=nthreads)
    nband = freq_bin_idx.size
    ref = np.zeros((nband, nx, ny), dtype=np.float64)
    for i in range(nband):
        ind = slice(freq_bin_idx[i], freq_bin_idx[i] + freq_bin_counts[i])
        ref[i] = explicit_gridder(uvw, freq[ind], vis[:, ind], wgt[:, ind],
                                  nx, ny, cell, cell, True)

    # l2 error should be within epsilon of zero
    assert_allclose(_l2error(image, ref), 0, atol=epsilon)


@pmp("nx", (30,))
@pmp("ny", (50, 128))
@pmp("fov", (0.5, 2.5))
@pmp("nrow", (333, 5000,))
@pmp("nchan", (1, 4))
@pmp("nband", (1, 2))
@pmp("precision", ('single', 'double'))
@pmp("nthreads", (6,))
def test_adjointness(nx, ny, fov, nrow, nchan, nband,
                     precision, nthreads):
    # instead of explicitly testing the degridder we can just check that
    # it is consistent with the gridder i.e.
    #
    #  <R.H y, x> = <y.H, Rx>
    #
    # where R.H is the gridder, R is the degridder and x and y are randomly
    # drawn image and visibilities respectively
    if nband > nchan:
        return
    from africanus.gridding.wgridder import dirty, model
    if precision == 'single':
        real_type = np.float32
        complex_type = np.complex64
        tol = 1e-4
    else:
        real_type = np.float64
        complex_type = np.complex128
        tol = 1e-12
    np.random.seed(420)
    cell = fov*np.pi/180/nx
    f0 = 1e9
    freq = (f0 + np.arange(nchan)*(f0/nchan))
    uvw = ((np.random.rand(nrow, 3)-0.5) /
           (cell*freq[-1]/lightspeed))
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
    image = dirty(uvw, freq, vis, freq_bin_idx, freq_bin_counts, nx, ny, cell,
                  weights=wgt, nthreads=nthreads)
    model_im = np.random.randn(nband, nx, ny).astype(real_type)
    modelvis = model(uvw, freq, model_im, freq_bin_idx, freq_bin_counts,
                     cell, weights=wgt, nthreads=nthreads)

    # should have relative tolerance close to machine precision
    assert_allclose(np.vdot(vis, modelvis).real, np.vdot(image, model_im),
                    rtol=tol)


@pmp("nx", (20, ))
@pmp("ny", (32, 70))
@pmp("fov", (1.5, 3.5))
@pmp("nrow", (222, 777,))
@pmp("nchan", (1, 5))
@pmp("nband", (1, 3))
@pmp("precision", ('single', 'double'))
@pmp("nthreads", (3,))
def test_residual(nx, ny, fov, nrow, nchan, nband,
                  precision, nthreads):
    # Compare the result of im2residim to
    #   VR = V - Rx   - computed with im2vis
    #   IR = R.H VR   - computed with vis2im
    from africanus.gridding.wgridder import dirty, model, residual
    np.random.seed(420)
    if precision == 'single':
        real_type = np.float32
        complex_type = np.complex64
        decimal = 4
    else:
        real_type = np.float64
        complex_type = np.complex128
        decimal = 12
    cell = fov*np.pi/180/nx
    f0 = 1e9
    freq = (f0 + np.arange(nchan)*(f0/nchan))
    uvw = ((np.random.rand(nrow, 3)-0.5) /
           (cell*freq[-1]/lightspeed))
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
    model_im = np.random.randn(nband, nx, ny).astype(real_type)
    modelvis = model(uvw, freq, model_im, freq_bin_idx, freq_bin_counts, cell,
                     nthreads=nthreads)
    residualvis = vis - modelvis
    residim1 = dirty(uvw, freq, residualvis, freq_bin_idx, freq_bin_counts,
                     nx, ny, cell, weights=wgt, nthreads=nthreads)

    residim2 = residual(uvw, freq, model_im, vis, freq_bin_idx,
                        freq_bin_counts, cell, weights=wgt,
                        nthreads=nthreads)

    # These are essentially computing the same thing just in a different
    # order so should be close to machine precision
    rmax = np.maximum(np.abs(residim1).max(), np.abs(residim2).max())
    assert_array_almost_equal(
        residim1/rmax, residim2/rmax, decimal=decimal)


@pmp("nx", (30, 250))
@pmp("ny", (128,))
@pmp("fov", (5.0,))
@pmp("nrow", (3333, 10000))
@pmp("nchan", (1, 8))
@pmp("nband", (1, 2))
@pmp("precision", ('single', 'double'))
@pmp("nthreads", (1, 4))
@pmp("nchunks", (1, 3))
def test_dask_dirty(nx, ny, fov, nrow, nchan, nband,
                    precision, nthreads, nchunks):
    da = pytest.importorskip("dask.array")
    from africanus.gridding.wgridder import dirty as dirty_np
    from africanus.gridding.wgridder.dask import dirty
    np.random.seed(420)
    if precision == 'single':
        real_type = np.float32
        complex_type = np.complex64
        decimal = 4  # does not pass at 5
    else:
        real_type = np.float64
        complex_type = np.complex128
        decimal = 7
    cell = fov*np.pi/180/nx
    f0 = 1e9
    freq = (f0 + np.arange(nchan)*(f0/nchan))
    uvw = ((np.random.rand(nrow, 3)-0.5) /
           (cell*freq[-1]/lightspeed))
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
    image = dirty_np(uvw, freq, vis, freq_bin_idx, freq_bin_counts, nx, ny,
                     cell, weights=wgt, nthreads=nthreads)

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

    image_da = dirty(uvw_da, freq_da, vis_da, freq_bin_idx_da,
                     freq_bin_counts_da, nx, ny, cell, weights=wgt_da,
                     nthreads=nthreads).compute()

    # relative error should agree to within epsilon
    dmax = np.maximum(np.abs(image).max(), np.abs(image_da).max())
    assert_array_almost_equal(image/dmax, image_da/dmax,
                              decimal=decimal)


@pmp("nx", (30, 250))
@pmp("ny", (128,))
@pmp("fov", (5.0,))
@pmp("nrow", (3333, 10000))
@pmp("nchan", (1, 8))
@pmp("nband", (1, 2))
@pmp("precision", ('single', 'double'))
@pmp("nthreads", (1, 4))
@pmp("nchunks", (1, 3))
def test_dask_model(nx, ny, fov, nrow, nchan, nband,
                    precision, nthreads, nchunks):
    da = pytest.importorskip("dask.array")
    from africanus.gridding.wgridder import model as model_np
    from africanus.gridding.wgridder.dask import model
    np.random.seed(420)
    if precision == 'single':
        real_type = np.float32
        complex_type = np.complex64
        decimal = 4  # does not pass at 5
    else:
        real_type = np.float64
        complex_type = np.complex128
        decimal = 7
    cell = fov*np.pi/180/nx
    f0 = 1e9
    freq = (f0 + np.arange(nchan)*(f0/nchan))
    uvw = ((np.random.rand(nrow, 3)-0.5) /
           (cell*freq[-1]/lightspeed))
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
    image = np.random.randn(nband, nx, ny).astype(real_type)

    vis = model_np(uvw, freq, image, freq_bin_idx, freq_bin_counts, cell,
                   weights=wgt, nthreads=nthreads)

    # now get result using dask
    rows_per_task = int(np.ceil(nrow/nchunks))
    row_chunks = (nchunks-1) * (rows_per_task,)
    row_chunks += (nrow - np.sum(row_chunks),)
    freq_da = da.from_array(freq, chunks=step)
    uvw_da = da.from_array(uvw, chunks=(row_chunks, -1))
    image_da = da.from_array(image, chunks=(1, nx, ny))
    wgt_da = da.from_array(wgt, chunks=(row_chunks, step))
    freq_bin_idx_da = da.from_array(freq_bin_idx, chunks=1)
    freq_bin_counts_da = da.from_array(freq_bin_counts, chunks=1)

    vis_da = model(uvw_da, freq_da, image_da, freq_bin_idx_da,
                   freq_bin_counts_da, cell, weights=wgt_da,
                   nthreads=nthreads).compute()

    # relative error should agree to within epsilon
    vmax = np.maximum(np.abs(vis).max(), np.abs(vis_da).max())
    assert_array_almost_equal(vis/vmax, vis_da/vmax,
                              decimal=decimal)


@pmp("nx", (30, 250))
@pmp("ny", (128,))
@pmp("fov", (5.0,))
@pmp("nrow", (3333, 10000))
@pmp("nchan", (1, 8))
@pmp("nband", (1, 2))
@pmp("precision", ('single', 'double'))
@pmp("nthreads", (1, 4))
@pmp("nchunks", (1, 3))
def test_dask_residual(nx, ny, fov, nrow, nchan, nband,
                       precision, nthreads, nchunks):
    da = pytest.importorskip("dask.array")
    from africanus.gridding.wgridder import residual as residual_np
    from africanus.gridding.wgridder.dask import residual
    np.random.seed(420)
    if precision == 'single':
        real_type = np.float32
        complex_type = np.complex64
        decimal = 4  # does not pass at 5
    else:
        real_type = np.float64
        complex_type = np.complex128
        decimal = 7
    cell = fov*np.pi/180/nx
    f0 = 1e9
    freq = (f0 + np.arange(nchan)*(f0/nchan))
    uvw = ((np.random.rand(nrow, 3)-0.5) /
           (cell*freq[-1]/lightspeed))
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
    image = np.random.randn(nband, nx, ny).astype(real_type)
    residim_np = residual_np(uvw, freq, image, vis, freq_bin_idx,
                             freq_bin_counts, cell, weights=wgt,
                             nthreads=nthreads)

    rows_per_task = int(np.ceil(nrow/nchunks))
    row_chunks = (nchunks-1) * (rows_per_task,)
    row_chunks += (nrow - np.sum(row_chunks),)
    freq_da = da.from_array(freq, chunks=step)
    uvw_da = da.from_array(uvw, chunks=(row_chunks, -1))
    image_da = da.from_array(image, chunks=(1, nx, ny))
    vis_da = da.from_array(vis, chunks=(row_chunks, step))
    wgt_da = da.from_array(wgt, chunks=(row_chunks, step))
    freq_bin_idx_da = da.from_array(freq_bin_idx, chunks=1)
    freq_bin_counts_da = da.from_array(freq_bin_counts, chunks=1)

    residim_da = residual(uvw_da, freq_da, image_da, vis_da,
                          freq_bin_idx_da, freq_bin_counts_da,
                          cell, weights=wgt_da, nthreads=nthreads).compute()

    # should agree to within epsilon
    rmax = np.maximum(np.abs(residim_np).max(), np.abs(residim_da).max())
    assert_array_almost_equal(
        residim_np/rmax, residim_da/rmax, decimal=decimal)
