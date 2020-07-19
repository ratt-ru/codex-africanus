# -*- coding: utf-8 -*-

import unittest
import numpy as np
from africanus.gridding.perleypolyhedron import kernels, gridder, degridder, policies
from africanus.dft.kernels import im_to_vis
import os

DEBUG = True

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
        
    def test_construct_kernels(self):
        import matplotlib
        matplotlib.use("agg")  
        from matplotlib import pyplot as plt
        plt.figure()
        WIDTH = 5
        OVERSAMP = 101
        l = kernels.uspace(WIDTH, OVERSAMP)
        sel = l <= WIDTH//2
        plt.axvline(0.5, 0, 1, ls="--", c="k")
        plt.axvline(-0.5, 0, 1, ls="--", c="k")
        plt.plot(l[sel]*OVERSAMP/2/np.pi, 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(kernels.kbsinc(WIDTH, oversample=OVERSAMP, order=0)[sel])))), label="kbsinc order 0")
        plt.plot(l[sel]*OVERSAMP/2/np.pi, 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(kernels.kbsinc(WIDTH, oversample=OVERSAMP, order=15)[sel])))), label="kbsinc order 15")
        plt.plot(l[sel]*OVERSAMP/2/np.pi, 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(kernels.hanningsinc(WIDTH, oversample=OVERSAMP)[sel])))), label="hanning sinc")
        plt.plot(l[sel]*OVERSAMP/2/np.pi, 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(kernels.sinc(WIDTH, oversample=OVERSAMP)[sel])))), label="sinc")
        plt.xlim(-10,10)
        plt.legend()
        plt.ylabel("Response [dB]")
        plt.xlabel("FoV")
        plt.grid(True)
        plt.savefig(os.path.join(os.environ.get("TMPDIR","/tmp"), "aakernels.png"))

    def test_taps(self):
        oversample=14
        W = 5
        taps = kernels.uspace(W, oversample=oversample)
        assert taps[oversample*(W//2)] == 0
        assert taps[0] == -(W // 2)
        assert taps[-oversample] == W // 2

    def test_packunpack(self):
        oversample=4
        W = 3
        K = kernels.uspace(W, oversample=oversample)
        Kp = kernels.pack_kernel(K, W, oversample=oversample)
        Kup = kernels.unpack_kernel(Kp, W, oversample=oversample)
        assert np.all(K == Kup)
        assert np.allclose(K, [-1.0,-0.75,-0.5,-0.25,0,
                               0.25, 0.5, 0.75, 1.0,
                               1.25, 1.5, 1.75])
        assert np.allclose(Kp, [-1.0,0,1.0,
                                -0.75,0.25,1.25,
                                -0.5,0.5,1.5,
                                -0.25,0.75,1.75])
    
    def test_grid(self):
        # construct kernel, lets use a fake kernel to check positioning
        W = 5
        OS = 3
        kern = kernels.pack_kernel(kernels.uspace(W, oversample=OS), W, oversample=OS)
        
        # offset 0
        uvw = np.array([[0,0,0]])
        vis = np.array([[[1.0+0j,1.0+0j]]])
        grid = gridder.gridder(uvw,vis,np.array([1.0]),np.array([0]),
                               64,30,(0,0),(0,0),kern,W,OS,"None","None","I_FROM_XXYY", "conv_1d_axisymmetric_packed_scatter")
        assert np.isclose(grid[0,64//2-2,64//2-2], 4.0)
        assert np.isclose(grid[0,64//2-1,64//2-1], 1.0)
        assert np.isclose(grid[0,64//2,64//2], 0.0)
        assert np.isclose(grid[0,64//2+1,64//2+1], +1.0)
        assert np.isclose(grid[0,64//2+2,64//2+2], +4.0)
        
        # offset 1
        scale = 64 * 30 / 3600.0 * np.pi / 180.0 / 1.0
        uvw = np.array([[0.4/scale,0,0]])
        vis = np.array([[[1.0+0j,1.0+0j]]])
        grid = gridder.gridder(uvw,vis,np.array([1.0]),np.array([0]),
                               64,30,(0,0),(0,0),kern,W,OS,"None","None","I_FROM_XXYY", "conv_1d_axisymmetric_packed_scatter")
        assert np.isclose(grid[0,64//2-2,64//2-2], 2*1.666666666666666)
        assert np.isclose(grid[0,64//2-1,64//2-1], 1*0.666666666666666)
        assert np.isclose(grid[0,64//2,64//2], 0.0*0.33333333333333333)
        assert np.isclose(grid[0,64//2+1,64//2+1], 1*1.33333333333333333)
        assert np.isclose(grid[0,64//2+2,64//2+2], 2*2.33333333333333333)

        # offset -1
        scale = 64 * 30 / 3600.0 * np.pi / 180.0 / 1.0
        uvw = np.array([[-0.22222/scale,0,0]])
        vis = np.array([[[1.0+0j,1.0+0j]]])
        grid = gridder.gridder(uvw,vis,np.array([1.0]),np.array([0]),
                               64,30,(0,0),(0,0),kern,W,OS,"None","None","I_FROM_XXYY", "conv_1d_axisymmetric_packed_scatter")
        assert np.isclose(grid[0,64//2-2,64//2-3], -2*-1.3333333333333333)
        assert np.isclose(grid[0,64//2-1,64//2-2], -1*-0.3333333333333333)
        assert np.isclose(grid[0,64//2,64//2-1], 0*0.666666666666666666)
        assert np.isclose(grid[0,64//2+1,64//2], 1*1.6666666666666666666)
        assert np.isclose(grid[0,64//2+2,64//2+1], 2*2.6666666666666666666)

    def test_degrid(self):
        # construct kernel, lets use a fake kernel to check positioning
        W = 5
        OS = 3
        kern = kernels.pack_kernel(kernels.uspace(W, oversample=OS), W, oversample=OS)
        
        # offset 0
        uvw = np.array([[0,0,0]])
        grid = np.ones((1,64,64), dtype=np.complex128)
        vis = degridder.degridder(uvw,grid,np.array([1.0]),np.array([0]),
                                  30,(0,0),(0,0),kern,W,OS,
                                  "None","None",
                                  "XXYY_FROM_I", "conv_1d_axisymmetric_packed_gather")
        assert np.isclose(vis[0,0,0], 0.0)
        

    def test_facetcodepath(self):
        # construct kernel
        W = 5
        OS = 3
        kern = kernels.pack_kernel(kernels.kbsinc(W, oversample=OS), W, oversample=OS)
        
        # offset 0
        uvw = np.array([[0,0,0]])
        vis = np.array([[[1.0+0j,1.0+0j]]])
        grid = gridder.gridder(uvw,vis,np.array([1.0]),np.array([0]),
                               64,30,(0,0),(0,0),kern,W,OS,"rotate","phase_rotate","I_FROM_XXYY", "conv_1d_axisymmetric_packed_scatter")

    def test_degrid_dft(self):
        # construct kernel
        W = 5
        OS = 3
        kern = kernels.pack_kernel(kernels.kbsinc(W, oversample=OS), W, oversample=OS)
        nrow = 1000
        # assume gaussian distribution for simulated uvw coordinates
        #uvw = np.random.normal(loc=0.0,scale=1000.0,size=(nrow, 3))
        uvw = np.column_stack((np.linspace(-50,50,1000),
                               np.linspace(-50,50,1000),
                               np.zeros(1000)))
        pxacrossbeam = 10
        wavelength = np.array([299792458.0/1.4e9])
        cell = np.rad2deg(wavelength[0]/(2*max(np.max(uvw[:,0]), np.max(uvw[:,1]))/pxacrossbeam))
        npix = 512
        mod = np.zeros((1, npix, npix), dtype=np.complex64)
        # # cross model
        # for n in np.arange(npix//2, npix, npix//5):
        #     mod[0, n, n] = 1.0
        #     mod[0, n, -n] = 1.0
        #     mod[0, -n, n] = 1.0
        #     mod[0, -n, -n] = 1.0
        mod[0,npix//4,npix//4] = 1.0

        ftmod = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(mod[0,:,:]))).reshape((1, npix, npix)) * npix**2
        chanmap = np.array([0])
        vis_degrid = degridder.degridder(uvw,
                                         ftmod,
                                         wavelength,
                                         chanmap,
                                         cell,
                                         (0, np.pi/4.0),
                                         (0, np.pi/4.0),
                                         kern,
                                         W,
                                         OS,
                                         "None", # no faceting
                                         "None", # no faceting
                                         "XXYY_FROM_I", 
                                         "conv_1d_axisymmetric_packed_gather")
        
        m, l = np.meshgrid(np.arange(-npix//2, npix//2) * np.deg2rad(cell),
                           np.arange(-npix//2, npix//2) * np.deg2rad(cell))
        lm = np.column_stack((l.flatten(),m.flatten()))
        vis_dft = im_to_vis(mod[0,:,:].reshape(1,1,npix*npix).T.copy(), uvw, lm, 1 / wavelength)
        import matplotlib
        matplotlib.use("agg")  
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(vis_degrid[:,0,0].real, label="$\Re(\mathtt{degrid})$")
        plt.plot(vis_dft[:,0,0].real, label="$\Re(\mathtt{dft})$")
        plt.legend()
        plt.xlabel("sample")
        plt.ylabel("Real of predicted")
        plt.savefig(os.path.join(os.environ.get("TMPDIR","/tmp"), "degrid_vs_dft.png"))
if __name__ == "__main__":
    unittest.main()