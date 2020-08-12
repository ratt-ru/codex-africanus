# -*- coding: utf-8 -*-

import unittest
import numpy as np
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from africanus.gridding.perleypolyhedron import kernels, gridder, degridder, policies
from africanus.gridding.perleypolyhedron import dask as dwrap
from africanus.dft.dask import im_to_vis, vis_to_im
from africanus.coordinates import radec_to_lmn
import os

class griddertest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        unittest.TestCase.setUpClass()

    @classmethod
    def tearDownClass(cls):
        unittest.TestCase.tearDownClass()

    def tearDown(self):
        unittest.TestCase.tearDown(self)

    def setUp(self):
        unittest.TestCase.setUp(self)

    def test_gridder_dask(self):
        # construct kernel
        W = 5
        OS = 9
        kern = kernels.pack_kernel(kernels.kbsinc(W, oversample=OS), W, OS)
        nrow = 5000
        np.random.seed(0)
        # simulate some ficticious baselines rotated by an hour angle
        row_chunks = nrow // 1000
        uvw = np.zeros((nrow, 3), dtype=np.float64)
        blpos = np.random.uniform(26, 10000, size=(25, 3))
        ntime = int(nrow / 25.0)
        d0 = np.pi/4.0
        for n in range(25):
            for ih0, h0 in enumerate(np.linspace(np.deg2rad(-20), np.deg2rad(20), ntime)):
                s = np.sin
                c = np.cos
                R = np.array([
                    [s(h0), c(h0), 0],
                    [-s(d0)*c(h0), s(d0)*s(h0), c(d0)],
                    [c(d0)*c(h0), -c(d0)*s(h0), s(d0)]
                ])
                uvw[n*ntime+ih0,:] = np.dot(R,blpos[n,:].T)
        uvw = da.from_array(uvw, chunks=(row_chunks, 3))
        pxacrossbeam = 5
        frequency = [1.4e9]
        wavelength = da.from_array([299792458.0/f for f in frequency], chunks=1)
        cell = da.rad2deg(wavelength[0]/(max(da.max(da.absolute(uvw[:,0])),
                                             da.max(da.absolute(uvw[:,1])))*pxacrossbeam))
        npix = 2048
        npixfacet = 100
        fftpad=1.1
        mod = da.ones((1, 1, 1), dtype=np.complex64)
        deltaradec = np.array([[600 * np.deg2rad(cell), 600 * np.deg2rad(cell)]])
        image_centres = deltaradec + np.array([[0, d0]])
        lm = da.from_array(radec_to_lmn(deltaradec + np.array([[0, d0]]), phase_centre=np.array([0, d0])))
        chanmap = da.from_array([0], chunks=1)
        detaper_facet = kernels.compute_detaper_dft_seperable(int(npixfacet*fftpad), kernels.unpack_kernel(kern, W, OS), W, OS)
        #vis_dft = im_to_vis(mod, uvw, lm[:,0:2], frequency).repeat(2, axis=1).reshape(nrow,1,2)
        vis_dft = da.ones(shape=(nrow,1,2), dtype=np.complex64, chunks=(row_chunks,1,2))
        vis_grid_facet = dwrap.gridder(uvw,
                                       vis_dft,
                                       wavelength,
                                       chanmap,
                                       int(npixfacet*fftpad),
                                       cell * 3600.0,
                                       image_centres,
                                       (0, d0),
                                       kern,
                                       W,
                                       OS,
                                       "rotate",
                                       "phase_rotate",
                                       "I_FROM_XXYY",
                                       "conv_1d_axisymmetric_packed_scatter",
                                       do_normalize=True)
        with ProgressBar():
            vis_grid_facet.compute(scheduler='single-threaded')
        # ftvisfacet = (np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(vis_grid_facet[0,:,:]))).reshape((1, int(npixfacet*fftpad), int(npixfacet*fftpad)))).real / detaper_facet * int(npixfacet*fftpad) ** 2
        # ftvisfacet = ftvisfacet[:,
        #               int(npixfacet*fftpad)//2-npixfacet//2:int(npixfacet*fftpad)//2-npixfacet//2+npixfacet,
        #               int(npixfacet*fftpad)//2-npixfacet//2:int(npixfacet*fftpad)//2-npixfacet//2+npixfacet]
        # from matplotlib import pyplot as plt
        # plt.figure()
        # plt.imshow(ftvisfacet)
        # plt.show()

if __name__ == "__main__":
    unittest.main()