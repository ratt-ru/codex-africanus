# -*- coding: utf-8 -*-

import unittest
import numpy as np
import dask
import dask.array as da
import time
from dask.diagnostics import ProgressBar
from africanus.gridding.perleypolyhedron import kernels, gridder, degridder, policies
from africanus.gridding.perleypolyhedron import dask as dwrap
from africanus.dft.dask import im_to_vis, vis_to_im
from africanus.coordinates import radec_to_lmn
import os

class clock:
    def __init__(self, identifier="untitled"):
        self._id = identifier
        self._elapsed = 0.0
        self._onenter = 0.0
        self._onexit = 0.0

    def __enter__(self):
        self._onenter = time.time() 
        return self

    def __exit__(self, extype, exval, tb):
        self._onexit = time.time()
        self._elapsed = self._onexit - self._onenter

    @property
    def elapsed(self):
        return self._elapsed
    
    def __str__(self):
        res = "{0:s}: Walltime {1:.0f}m{2:.2f}s elapsed".format(self._id,
                                                                self.elapsed // 60,
                                                                self.elapsed - (self.elapsed // 60)*60)
        return res

    __repr__ = __str__

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
        with clock("DASK gridding") as tictoc:
            # construct kernel
            W = 5
            OS = 9
            kern = kernels.pack_kernel(kernels.kbsinc(W, oversample=OS), W, OS)
            nrow = int(1e6)
            np.random.seed(0)
            # simulate some ficticious baselines rotated by an hour angle
            row_chunks = nrow // 10
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
            nchan = 128
            frequency = da.from_array(np.linspace(1.0e9,1.4e9,nchan), chunks=(nchan,))
            wavelength = 299792458.0/frequency
            cell = da.rad2deg(wavelength[0]/(max(da.max(da.absolute(uvw[:,0])),
                                                da.max(da.absolute(uvw[:,1])))*pxacrossbeam))
            npix = 2048
            npixfacet = 100
            fftpad=1.1
            
            image_centres = da.from_array(np.array([[0, d0]]), chunks=(1,2))
            chanmap = da.from_array(np.zeros(nchan, dtype=np.int64), chunks=(nchan,))
            detaper_facet = kernels.compute_detaper_dft_seperable(int(npixfacet*fftpad), kernels.unpack_kernel(kern, W, OS), W, OS)
            vis_dft = da.ones(shape=(nrow,nchan,2), chunks=(row_chunks,nchan,2), dtype=np.complex64)
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
                                        "None",
                                        "None",
                                        "I_FROM_XXYY",
                                        "conv_1d_axisymmetric_packed_scatter",
                                        do_normalize=True)
            with ProgressBar():
               vis_grid_facet = vis_grid_facet.compute()
            ftvisfacet = (np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(vis_grid_facet[0,:,:]))).reshape((1, int(npixfacet*fftpad), int(npixfacet*fftpad)))).real / detaper_facet * int(npixfacet*fftpad) ** 2
            ftvisfacet = ftvisfacet[:,
                        int(npixfacet*fftpad)//2-npixfacet//2:int(npixfacet*fftpad)//2-npixfacet//2+npixfacet,
                        int(npixfacet*fftpad)//2-npixfacet//2:int(npixfacet*fftpad)//2-npixfacet//2+npixfacet]
        print(tictoc)
        assert(np.abs(np.max(ftvisfacet[0,:,:]) - 1.0) < 1.0e-6) 

    def test_gridder_nondask(self):
        with clock("Non-DASK gridding") as tictoc:
            # construct kernel
            W = 5
            OS = 9
            kern = kernels.pack_kernel(kernels.kbsinc(W, oversample=OS), W, OS)
            nrow = int(1e6)
            np.random.seed(0)
            # simulate some ficticious baselines rotated by an hour angle
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
            pxacrossbeam = 5
            nchan = 128
            frequency = np.linspace(1.0e9,1.4e9,nchan)
            wavelength = 299792458.0/frequency
            cell = np.rad2deg(wavelength[0]/(max(np.max(np.absolute(uvw[:,0])),
                                                 np.max(np.absolute(uvw[:,1])))*pxacrossbeam))
            npix = 2048
            npixfacet = 100
            fftpad=1.1
            
            image_centres = np.array([[0, d0]])
            chanmap = np.zeros(nchan, dtype=np.int64)
            detaper_facet = kernels.compute_detaper_dft_seperable(int(npixfacet*fftpad), kernels.unpack_kernel(kern, W, OS), W, OS)
            vis_dft = np.ones((nrow,nchan,2), dtype=np.complex64)
            vis_grid_facet = gridder.gridder(uvw,
                                        vis_dft,
                                        wavelength,
                                        chanmap,
                                        int(npixfacet*fftpad),
                                        cell * 3600.0,
                                        image_centres[0,:],
                                        (0, d0),
                                        kern,
                                        W,
                                        OS,
                                        "None",
                                        "None",
                                        "I_FROM_XXYY",
                                        "conv_1d_axisymmetric_packed_scatter",
                                        do_normalize=True)
            ftvisfacet = (np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(vis_grid_facet[0,:,:]))).reshape((1, int(npixfacet*fftpad), int(npixfacet*fftpad)))).real / detaper_facet * int(npixfacet*fftpad) ** 2
            ftvisfacet = ftvisfacet[:,
                        int(npixfacet*fftpad)//2-npixfacet//2:int(npixfacet*fftpad)//2-npixfacet//2+npixfacet,
                        int(npixfacet*fftpad)//2-npixfacet//2:int(npixfacet*fftpad)//2-npixfacet//2+npixfacet]
        print(tictoc)
        assert(np.abs(np.max(ftvisfacet[0,:,:]) - 1.0) < 1.0e-6)

if __name__ == "__main__":
    unittest.main()