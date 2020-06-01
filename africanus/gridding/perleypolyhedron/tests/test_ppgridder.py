# -*- coding: utf-8 -*-

import unittest
import numpy as np
from africanus.gridding.perleypolyhedron import kernels, gridder, policies
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
                               64,30,(0,0),(0,0),kern,W,OS,"None","None","I_FROM_XXYY", "conv_1d_axisymmetric_packed")
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
                               64,30,(0,0),(0,0),kern,W,OS,"None","None","I_FROM_XXYY", "conv_1d_axisymmetric_packed")
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
                               64,30,(0,0),(0,0),kern,W,OS,"None","None","I_FROM_XXYY", "conv_1d_axisymmetric_packed")
        assert np.isclose(grid[0,64//2-2,64//2-3], -2*-1.3333333333333333)
        assert np.isclose(grid[0,64//2-1,64//2-2], -1*-0.3333333333333333)
        assert np.isclose(grid[0,64//2,64//2-1], 0*0.666666666666666666)
        assert np.isclose(grid[0,64//2+1,64//2], 1*1.6666666666666666666)
        assert np.isclose(grid[0,64//2+2,64//2+1], 2*2.6666666666666666666)
        
if __name__ == "__main__":
    unittest.main()