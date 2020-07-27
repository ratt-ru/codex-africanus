# -*- coding: utf-8 -*-

import unittest
import numpy as np
from africanus.gridding.perleypolyhedron import kernels, gridder, degridder, policies
from africanus.dft.kernels import im_to_vis, vis_to_im
from africanus.coordinates import radec_to_lmn
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
        
    # def test_construct_kernels(self):
    #     import matplotlib
    #     matplotlib.use("agg")  
    #     from matplotlib import pyplot as plt
    #     plt.figure()
    #     WIDTH = 5
    #     OVERSAMP = 101
    #     l = kernels.uspace(WIDTH, OVERSAMP)
    #     sel = l <= WIDTH//2
    #     plt.axvline(0.5, 0, 1, ls="--", c="k")
    #     plt.axvline(-0.5, 0, 1, ls="--", c="k")
    #     plt.plot(l[sel]*OVERSAMP/2/np.pi, 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(kernels.kbsinc(WIDTH, oversample=OVERSAMP, order=0)[sel])))), label="kbsinc order 0")
    #     plt.plot(l[sel]*OVERSAMP/2/np.pi, 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(kernels.kbsinc(WIDTH, oversample=OVERSAMP, order=15)[sel])))), label="kbsinc order 15")
    #     plt.plot(l[sel]*OVERSAMP/2/np.pi, 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(kernels.hanningsinc(WIDTH, oversample=OVERSAMP)[sel])))), label="hanning sinc")
    #     plt.plot(l[sel]*OVERSAMP/2/np.pi, 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(kernels.sinc(WIDTH, oversample=OVERSAMP)[sel])))), label="sinc")
    #     plt.xlim(-10,10)
    #     plt.legend()
    #     plt.ylabel("Response [dB]")
    #     plt.xlabel("FoV")
    #     plt.grid(True)
    #     plt.savefig(os.path.join(os.environ.get("TMPDIR","/tmp"), "aakernels.png"))

    # def test_taps(self):
    #     oversample=14
    #     W = 5
    #     taps = kernels.uspace(W, oversample=oversample)
    #     assert taps[oversample*(W//2)] == 0
    #     assert taps[0] == -(W // 2)
    #     assert taps[-oversample] == W // 2

    # def test_packunpack(self):
    #     oversample=4
    #     W = 3
    #     K = kernels.uspace(W, oversample=oversample)
    #     Kp = kernels.pack_kernel(K, W, oversample=oversample)
    #     Kup = kernels.unpack_kernel(Kp, W, oversample=oversample)
    #     assert np.all(K == Kup)
    #     assert np.allclose(K, [-1.0,-0.75,-0.5,-0.25,0,
    #                            0.25, 0.5, 0.75, 1.0,
    #                            1.25, 1.5, 1.75])
    #     assert np.allclose(Kp, [-1.0,0,1.0,
    #                             -0.75,0.25,1.25,
    #                             -0.5,0.5,1.5,
    #                             -0.25,0.75,1.75])
    
    # def test_grid(self):
    #     # construct kernel, lets use a fake kernel to check positioning
    #     W = 5
    #     OS = 3
    #     kern = kernels.pack_kernel(kernels.uspace(W, oversample=OS), W, oversample=OS)
        
    #     # offset 0
    #     uvw = np.array([[0,0,0]])
    #     vis = np.array([[[1.0+0j,1.0+0j]]])
    #     grid = gridder.gridder(uvw,vis,np.array([1.0]),np.array([0]),
    #                            64,30,(0,0),(0,0),kern,W,OS,"None","None","I_FROM_XXYY", "conv_1d_axisymmetric_packed_scatter")
    #     assert np.isclose(grid[0,64//2-2,64//2-2], 4.0)
    #     assert np.isclose(grid[0,64//2-1,64//2-1], 1.0)
    #     assert np.isclose(grid[0,64//2,64//2], 0.0)
    #     assert np.isclose(grid[0,64//2+1,64//2+1], +1.0)
    #     assert np.isclose(grid[0,64//2+2,64//2+2], +4.0)
        
    #     # offset 1
    #     scale = 64 * 30 / 3600.0 * np.pi / 180.0 / 1.0
    #     uvw = np.array([[0.4/scale,0,0]])
    #     vis = np.array([[[1.0+0j,1.0+0j]]])
    #     grid = gridder.gridder(uvw,vis,np.array([1.0]),np.array([0]),
    #                            64,30,(0,0),(0,0),kern,W,OS,"None","None","I_FROM_XXYY", "conv_1d_axisymmetric_packed_scatter")
        
    #     assert np.isclose(grid[0,64//2-2,64//2-2], 2*1.666666666666666)
    #     assert np.isclose(grid[0,64//2-1,64//2-1], 1*0.666666666666666)
    #     assert np.isclose(grid[0,64//2,64//2], 0.0*0.33333333333333333)
    #     assert np.isclose(grid[0,64//2+1,64//2+1], 1*1.33333333333333333)
    #     assert np.isclose(grid[0,64//2+2,64//2+2], 2*2.33333333333333333)

    #     # offset -1
    #     scale = 64 * 30 / 3600.0 * np.pi / 180.0 / 1.0
    #     uvw = np.array([[-0.66666/scale,0,0]])
    #     vis = np.array([[[1.0+0j,1.0+0j]]])
    #     grid = gridder.gridder(uvw,vis,np.array([1.0]),np.array([0]),
    #                            64,30,(0,0),(0,0),kern,W,OS,"None","None","I_FROM_XXYY", "conv_1d_axisymmetric_packed_scatter")
    #     assert np.isclose(grid[0,64//2-2,64//2-3], -2*-1.66666666666666666)
    #     assert np.isclose(grid[0,64//2-1,64//2-2], -1*-0.66666666666666666)
    #     assert np.isclose(grid[0,64//2,64//2-1], 0*0.33333333333333333333)
    #     assert np.isclose(grid[0,64//2+1,64//2], 1*1.33333333333333333333)
    #     assert np.isclose(grid[0,64//2+2,64//2+1], 2*2.3333333333333333333)

    # def test_degrid(self):
    #     # construct kernel, lets use a fake kernel to check positioning
    #     W = 5
    #     OS = 3
    #     kern = kernels.pack_kernel(kernels.uspace(W, oversample=OS), W, oversample=OS)
        
    #     # offset 0
    #     uvw = np.array([[0,0,0]])
    #     grid = np.ones((1,64,64), dtype=np.complex128)
    #     vis = degridder.degridder(uvw,grid,np.array([1.0]),np.array([0]),
    #                               30,(0,0),(0,0),kern,W,OS,
    #                               "None","None",
    #                               "XXYY_FROM_I", "conv_1d_axisymmetric_packed_gather")
    #     assert np.isclose(vis[0,0,0], 0.0)
        

    # def test_facetcodepath(self):
    #     # construct kernel
    #     W = 5
    #     OS = 3
    #     kern = kernels.pack_kernel(kernels.kbsinc(W, oversample=OS), W, oversample=OS)
        
    #     # offset 0
    #     uvw = np.array([[0,0,0]])
    #     vis = np.array([[[1.0+0j,1.0+0j]]])
    #     grid = gridder.gridder(uvw,vis,np.array([1.0]),np.array([0]),
    #                            64,30,(0,0),(0,0),kern,W,OS,"rotate","phase_rotate","I_FROM_XXYY", "conv_1d_axisymmetric_packed_scatter")

    # def test_degrid_dft(self):
    #     # construct kernel
    #     W = 5
    #     OS = 3
    #     kern = kernels.pack_kernel(kernels.kbsinc(W, oversample=OS), W, oversample=OS)
    #     nrow = 200
    #     uvw = np.column_stack((5000.0 * np.cos(np.linspace(0,2*np.pi,1000)),
    #                            5000.0 * np.sin(np.linspace(0,2*np.pi,1000)),
    #                            np.zeros(1000)))
        
    #     pxacrossbeam = 10
    #     frequency = np.array([1.4e9])
    #     wavelength = np.array([299792458.0/f for f in frequency])

    #     cell = np.rad2deg(wavelength[0]/(2*max(np.max(np.abs(uvw[:,0])), 
    #                                            np.max(np.abs(uvw[:,1])))*pxacrossbeam))
    #     npix = 512
    #     mod = np.zeros((1, npix, npix), dtype=np.complex64)
    #     mod[0, npix//2 - 5, npix//2 -5] = 1.0

    #     ftmod = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(mod[0,:,:]))).reshape((1, npix, npix))
    #     chanmap = np.array([0])
    #     vis_degrid = degridder.degridder(uvw,
    #                                      ftmod,
    #                                      wavelength,
    #                                      chanmap,
    #                                      cell * 3600.0,
    #                                      (0, np.pi/4.0),
    #                                      (0, np.pi/4.0),
    #                                      kern,
    #                                      W,
    #                                      OS,
    #                                      "None", # no faceting
    #                                      "None", # no faceting
    #                                      "XXYY_FROM_I", 
    #                                      "conv_1d_axisymmetric_packed_gather")
        
    #     dec, ra = np.meshgrid(np.arange(-npix//2, npix//2) * np.deg2rad(cell),
    #                           np.arange(-npix//2, npix//2) * np.deg2rad(cell))
    #     radec = np.column_stack((ra.flatten(), dec.flatten()))
        
    #     vis_dft = im_to_vis(mod[0,:,:].reshape(1,1,npix*npix).T.copy(), uvw, radec, frequency)
        
    #     import matplotlib
    #     matplotlib.use("agg")  
    #     from matplotlib import pyplot as plt
    #     plt.figure()
    #     plt.plot(vis_degrid[:,0,0].real, label="$\Re(\mathtt{degrid})$")
    #     plt.plot(vis_dft[:,0,0].real, label="$\Re(\mathtt{dft})$")
    #     plt.plot(np.abs(vis_dft[:,0,0].real - vis_degrid[:,0,0].real), label="Error")
    #     plt.legend()
    #     plt.xlabel("sample")
    #     plt.ylabel("Real of predicted")
    #     plt.savefig(os.path.join(os.environ.get("TMPDIR","/tmp"), "degrid_vs_dft_re.png"))
    #     plt.figure()
    #     plt.plot(vis_degrid[:,0,0].imag, label="$\Im(\mathtt{degrid})$")
    #     plt.plot(vis_dft[:,0,0].imag, label="$\Im(\mathtt{dft})$")
    #     plt.plot(np.abs(vis_dft[:,0,0].imag - vis_degrid[:,0,0].imag), label="Error")
    #     plt.legend()
    #     plt.xlabel("sample")
    #     plt.ylabel("Real of predicted")
    #     plt.savefig(os.path.join(os.environ.get("TMPDIR","/tmp"), "degrid_vs_dft_im.png"))
    #     assert np.percentile(np.abs(vis_dft[:,0,0].real - vis_degrid[:,0,0].real),90.0) < 0.05
    #     assert np.percentile(np.abs(vis_dft[:,0,0].imag - vis_degrid[:,0,0].imag),90.0) < 0.05

    def test_grid_dft(self):
        # construct kernel
        W = 5
        OS = 3
        kern = kernels.pack_kernel(kernels.kbsinc(W, oversample=OS), W, oversample=OS)
        
        nrow = 1000
        uvw = np.random.normal(scale=6000, size=(nrow, 3))
        uvw[:,2] = 0.0 # ignore widefield effects for now

        pxacrossbeam = 10
        frequency = np.array([1.4e9])
        wavelength = np.array([299792458.0/f for f in frequency])

        cell = np.rad2deg(wavelength[0]/(2*max(np.max(np.abs(uvw[:,0])), 
                                               np.max(np.abs(uvw[:,1])))*pxacrossbeam))
        npix = 512
        mod = np.zeros((1, npix, npix), dtype=np.complex64)
        for n in [[int(n) for n in np.linspace(npix//10, npix//2-1, 5)][3]]:
           mod[0, npix//2 + n, npix//2 + n] = 1.0
           #mod[0, npix//2 + n, npix//2 - n] = 1.0
           #mod[0, npix//2 - n, npix//2 - n] = 1.0
           #mod[0, npix//2 - n, npix//2 + n] = 1.0
        
        dec, ra = np.meshgrid(np.arange(-npix//2, npix//2) * np.deg2rad(cell),
                              np.arange(-npix//2, npix//2) * np.deg2rad(cell))
        radec = np.column_stack((ra.flatten(), dec.flatten()))
        
        vis_dft = im_to_vis(mod[0,:,:].reshape(1,1,npix*npix).T.copy(), uvw, radec, frequency).repeat(2).reshape(nrow,1,2)   
        chanmap = np.array([0])
        taper_grid = gridder.gridder(np.zeros((1,3)),
                                     np.ones((1,1,2), dtype=np.complex64),
                                     wavelength,
                                     chanmap,
                                     npix,
                                     cell * 3600.0,
                                     (0, np.pi/4.0),
                                     (0, np.pi/4.0),
                                     kern,
                                     W,
                                     OS,
                                     "None", # no faceting
                                     "None", # no faceting
                                     "I_FROM_XXYY",
                                     "conv_1d_axisymmetric_packed_scatter",
                                     do_normalize=True)
        detaper = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(taper_grid[0,:,:]))).reshape((1, npix, npix))
        nterm = np.sqrt(1 - radec[:,0]**2 - radec[:,1]**2).reshape((npix, npix))
        vis_grid = gridder.gridder(uvw,
                                   vis_dft,
                                   wavelength,
                                   chanmap,
                                   npix,
                                   cell * 3600.0,
                                   (0, np.pi/4.0),
                                   (0, np.pi/4.0),
                                   kern,
                                   W,
                                   OS,
                                   "None", # no faceting
                                   "None", # no faceting
                                   "I_FROM_XXYY",
                                   "conv_1d_axisymmetric_packed_scatter",
                                   do_normalize=True)
        import ipdb; ipdb.set_trace()
        ftvis = (np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(vis_grid[0,:,:]))).reshape((1, npix, npix)) / detaper / nterm).real * npix ** 2 
        #dftvis = vis_to_im(vis_dft, uvw, radec, frequency, np.zeros(vis_dft.shape, dtype=np.bool)).T.copy().reshape(2,1,npix,npix) / npix
        #import matplotlib
        #matplotlib.use("agg")  
        from matplotlib import pyplot as plt
        plt.figure()
        #plt.subplot(131)
        plt.title("FFT")
        plt.imshow(ftvis[0,:,:])
        plt.colorbar()
        #plt.subplot(132)
        #plt.title("DFT")
        #plt.imshow(dftvis[0,0,:,:])
        #plt.colorbar()
        #plt.subplot(133)
        #plt.title("ABS diff")
        #plt.imshow(np.abs(ftvis[0,:,:] - dftvis[0,0,:,:]))
        #plt.colorbar()
        #plt.savefig(os.path.join(os.environ.get("TMPDIR","/tmp"), "grid_diff_dft.png"))
        plt.show()

        import ipdb; ipdb.set_trace()

    # def test_adjoint_ops(self):
    #     # test adjointness of gridding and degridding operators
    #     # construct kernel
    #     W = 5
    #     OS = 3
    #     kern = kernels.pack_kernel(kernels.kbsinc(W, oversample=OS), W, oversample=OS)
    #     nrow = 1000
    #     uvw = np.column_stack((5000.0 * np.cos(np.linspace(0,2*np.pi,1000)),
    #                            5000.0 * np.sin(np.linspace(0,2*np.pi,1000)),
    #                            np.zeros(1000)))
        
    #     pxacrossbeam = 10
    #     frequency = np.array([1.4e9])
    #     wavelength = np.array([299792458.0/f for f in frequency])

    #     cell = np.rad2deg(wavelength[0]/(2*max(np.max(np.abs(uvw[:,0])), 
    #                                            np.max(np.abs(uvw[:,1])))*pxacrossbeam))
    #     npix = 512
    #     mod = np.zeros((1, npix, npix), dtype=np.complex64)
    #     mod[0, npix//2 - 5, npix//2 -5] = 1.0

    #     ftmod = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(mod[0,:,:]))).reshape((1, npix, npix))
    #     chanmap = np.array([0])
    #     vis_degrid = degridder.degridder(uvw,
    #                                      ftmod,
    #                                      wavelength,
    #                                      chanmap,
    #                                      cell * 3600.0,
    #                                      (0, np.pi/4.0),
    #                                      (0, np.pi/4.0),
    #                                      kern,
    #                                      W,
    #                                      OS,
    #                                      "None", # no faceting
    #                                      "None", # no faceting
    #                                      "XXYY_FROM_I", 
    #                                      "conv_1d_axisymmetric_packed_gather")

    #     y = np.random.uniform(-5,+5,vis_degrid.shape)
    #     vis_grid = gridder.gridder(uvw,
    #                                y,
    #                                wavelength,
    #                                chanmap,
    #                                npix,
    #                                cell * 3600.0,
    #                                (0, np.pi/4.0),
    #                                (0, np.pi/4.0),
    #                                kern,
    #                                W,
    #                                OS,
    #                                "None", # no faceting
    #                                "None", # no faceting
    #                                "I_FROM_XXYY", 
    #                                "conv_1d_axisymmetric_packed_scatter")

    #     # test y* . vis_degrid(img) == grid(y) . ft(img) 
    #     lhs = np.dot(y.flatten(), vis_degrid.flatten()).real
    #     rhs = np.dot(vis_grid.flatten(), ftmod.flatten()).real

if __name__ == "__main__":
    unittest.main()