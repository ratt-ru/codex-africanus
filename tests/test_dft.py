#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

import numpy as np

#import pytest

def test_im_to_vis():
    """
    The simplest test here is to see if a single source at the phase centre
    returns simply the flux everywhere with zero imaginary part
    :return: 
    """
    from africanus.dft.kernels import im_to_vis
    #np.random.seed(123)

    Nrow = 100
    uvw = np.random.random(size=(Nrow, 3))
    Npix = 35  # must be odd for this test to work
    x = np.linspace(-0.1, 0.1, Npix)
    ll, mm = np.meshgrid(x, x)
    lm = np.vstack((ll.flatten(), mm.flatten())).T
    Nchan = 11
    frequency = np.linspace(1.0, 2.0, Nchan, endpoint=True)

    image = np.zeros([Npix, Npix, Nchan], dtype=np.float64)
    I0 = 1.0
    ref_freq = frequency[Nchan//2]
    Inu = I0*(frequency/ref_freq)**(-0.7)
    image[Npix//2, Npix//2, :] = Inu
    image = image.reshape(Npix**2, Nchan)

    vis = im_to_vis(image, uvw, lm, frequency)

    for i in xrange(Nchan):
        tmp = vis[:, i] - Inu[i]
        assert np.all(tmp.real < 1e-13)
        assert np.all(tmp.imag < 1e-13)


def test_vis_to_im():
    """
    Still thinking of a better test here but we can do here but the simplest test 
    does exactly the same as the above. If we have an auto-correlation we expect 
    to measure a flat image with value wsum
    :return: 
    """
    from africanus.dft.kernels import vis_to_im
    Nchan = 11

    vis = np.ones([1, Nchan], dtype=np.complex128)
    uvw = np.zeros([1, 3], dtype=np.float64)
    Npix = 5
    x = np.linspace(-0.1, 0.1, Npix)
    ll, mm = np.meshgrid(x, x)
    lm = np.vstack((ll.flatten(), mm.flatten())).T
    wsum = 1.0

    frequency = np.linspace(1.0, 2.0, Nchan, endpoint=True)

    image = vis_to_im(vis, uvw, lm, frequency)

    for i in xrange(Nchan):
        assert np.all(image[:, i] == wsum)


def test_adjointness():
    """
    She is the mother of all tests. The DFT should be perfectly self adjoint up to 
    machine precision. 
    :return: 
    """
    from africanus.dft.kernels import im_to_vis as R
    from africanus.dft.kernels import vis_to_im as RH

    np.random.seed(123)
    Npix = 33
    Nvis = 1000
    Nchan = 1

    uvw = np.random.random(size=(Nvis,3))
    x = np.linspace(-0.1, 0.1, Npix)
    ll, mm = np.meshgrid(x, x)
    lm = np.vstack((ll.flatten(), mm.flatten())).T
    frequency = np.array([1.0])

    gamma1 = np.random.randn(Npix**2, Nchan)
    gamma2 = np.random.randn(Nvis, Nchan)

    LHS = gamma2.T.dot(R(gamma1, uvw, lm, frequency))
    RHS = RH(gamma2, uvw, lm, frequency).T.dot(gamma1)

    assert np.all(np.abs(LHS - RHS) < 1e-5)

# if __name__=="__main__":
#     test_im_to_vis()
#     test_vis_to_im()
#     test_adjointness()
