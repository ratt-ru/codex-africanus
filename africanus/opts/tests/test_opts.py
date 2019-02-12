#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

import dask.array as da
import numpy as np

import pytest


def test_power_method():
    """
    test the operation of the power method which approximates the largest eigenvalue of the matrix,
    both methods should result in the same value.
    """
    from africanus.opts.sub_opts import pow_method as pm

    eig = 0

    while eig is 0:
        A = np.random.randn(10, 10)
        G = A.T.dot(A)
        eig_vals = np.linalg.eigvals(G)
        if min(eig_vals) > 0:
            eig = max(eig_vals)

    spec = pm(G.dot, G.conj().T.dot, [10, 1])

    assert (abs(eig-spec) < 1e-8)


def test_l2_projection():
    """
    Test that the l2 projection projects x onto a ball of radius eps around position y
    """
    from africanus.opts.sub_opts import da_proj_l2ball
    np.random.seed(111)

    x = da.random.random(10, chunks=10)
    y = da.random.random(10, chunks=10)

    eps = 1e-2

    projection = da_proj_l2ball(x, eps, y)

    assert np.all(abs(projection - y) <= eps)


def test_l1_projection():
    """
    Test that the l1 projection suppresses x by tau or sets it to zero if less than tau
    """
    from africanus.opts.sub_opts import da_proj_l1_plus_pos
    np.random.seed(111)

    x = da.random.random(10, chunks=10)
    tau = 0.5

    proj = da_proj_l1_plus_pos(x, tau)

    assert np.all(proj <= abs(x-tau))


def test_primal_dual_PSF():
    """
    Test the PSF primal dual on test data, this could get pricey
    """
    from africanus.opts.primaldual import primal_dual_solver
    from africanus.reduction.psf_redux import PSF_adjoint, PSF_response, make_dim_reduce_ops, whiten_noise, iFFT
    from africanus.opts.data_reader import data_reader
    from matplotlib import pyplot as plt

    data_path = "/home/antonio/Documents/Masters/Helpful_Stuff/WSCMSSSMFTestSuite/SSMF.MS_p0"
    ra_dec = np.array([[3.15126500e-05], [-0.00551471375]])

    uvw_dask, lm_dask, lm_pad_dask, frequency_dask, weights_dask, vis_dask, padding = data_reader(data_path, ra_dec, nrow=1000)

    wsum = da.sum(weights_dask)
    pad_pix = int(da.sqrt(lm_pad_dask.shape[0]))

    vis_grid, PSF_hat, Sigma_hat = make_dim_reduce_ops(uvw_dask, lm_pad_dask, frequency_dask, vis_dask, weights_dask)
    white_vis, white_psf_hat = whiten_noise(vis_grid, PSF_hat, Sigma_hat)
    dirty = da.absolute(iFFT(white_vis))#/da.absolute(da.sqrt(da.sum(Sigma_hat)))

    PSF_op = lambda image: PSF_response(image, white_psf_hat)
    PSF_adj = lambda vis: PSF_adjoint(vis, white_psf_hat)/wsum

    start = np.zeros_like(dirty, dtype=np.float64)
    start[pad_pix // 2, pad_pix // 2] = 10

    cleaned = primal_dual_solver(start, white_vis, PSF_op, PSF_adj, dask=False)

    plt.figure('ID')
    plt.imshow(dirty[padding:-padding, padding:-padding])
    plt.colorbar()

    plt.figure('IM')
    plt.imshow(cleaned[padding:-padding, padding:-padding]/da.absolute(da.sqrt(da.sum(Sigma_hat))))
    plt.colorbar()

    plt.show()


def test_primal_dual_DFT():
    from africanus.opts.primaldual import primal_dual_solver
    from africanus.opts.data_reader import data_reader
    from africanus.dft.dask import im_to_vis, vis_to_im
    from matplotlib import pyplot as plt

    data_path = "/home/antonio/Documents/Masters/Helpful_Stuff/WSCMSSSMFTestSuite/SSMF.MS_p0"
    ra_dec = np.array([[3.15126500e-05], [-0.00551471375]])

    uvw_dask, lm_dask, lm_pad_dask, frequency_dask, weights_dask, vis_dask, padding = data_reader(data_path, ra_dec)

    wsum = da.sum(weights_dask)
    npix = int(da.sqrt(lm_dask.shape[0]))

    operator = lambda i: im_to_vis(i, uvw_dask, lm_dask, frequency_dask)
    adjoint = lambda v: vis_to_im(v, uvw_dask, lm_dask, frequency_dask) / da.sqrt(wsum)

    start = np.zeros([npix**2, 1])
    start[npix**2//2, 0] = 10

    cleaned = primal_dual_solver(start, vis_dask, operator, adjoint)

    plt.figure('ID')
    dirty = adjoint(vis_dask).reshape([npix, npix])
    plt.imshow(dirty / da.sqrt(wsum))
    plt.colorbar()

    plt.figure('IM')
    plt.imshow(cleaned)
    plt.colorbar()

    plt.show()

if __name__=="__main__":
    test_primal_dual_PSF()
