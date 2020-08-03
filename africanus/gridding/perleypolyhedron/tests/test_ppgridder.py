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
        
    def test_construct_kernels(self):
        import matplotlib
        matplotlib.use("agg")  
        from matplotlib import pyplot as plt
        plt.figure()
        WIDTH = 5
        OVERSAMP = 101
        l = kernels.uspace(WIDTH, OVERSAMP)
        sel = l <= (WIDTH+2)//2
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
        assert taps[oversample*((W+2)//2)] == 0
        assert taps[0] == -((W+2) // 2)
        assert taps[-oversample] == (W+2) // 2

    def test_packunpack(self):
        oversample=4
        W = 3
        K = kernels.uspace(W, oversample=oversample)
        Kp = kernels.pack_kernel(K, W, oversample=oversample)
        Kup = kernels.unpack_kernel(Kp, W, oversample=oversample)
        assert np.all(K == Kup)
        assert np.allclose(K, [-2.0,-1.75,-1.5,-1.25,
                               -1.0,-0.75,-0.5,-0.25,0,
                               0.25, 0.5, 0.75, 1.0,
                               1.25, 1.5, 1.75, 2.0,
                               2.25, 2.5, 2.75])
        assert np.allclose(Kp, [-2.0,-1.0,0,1.0,2.0,
                                -1.75,-0.75,0.25,1.25,2.25,
                                -1.5,-0.5,0.5,1.5,2.5,
                                -1.25,-0.25,0.75,1.75,2.75])

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
        kern = kernels.kbsinc(W, oversample=OS)
        nrow = 200
        uvw = np.column_stack((5000.0 * np.cos(np.linspace(0,2*np.pi,1000)),
                               5000.0 * np.sin(np.linspace(0,2*np.pi,1000)),
                               np.zeros(1000)))
        
        pxacrossbeam = 10
        frequency = np.array([1.4e9])
        wavelength = np.array([299792458.0/f for f in frequency])

        cell = np.rad2deg(wavelength[0]/(2*max(np.max(np.abs(uvw[:,0])), 
                                               np.max(np.abs(uvw[:,1])))*pxacrossbeam))
        npix = 512
        mod = np.zeros((1, npix, npix), dtype=np.complex64)
        mod[0, npix//2 - 5, npix//2 -5] = 1.0

        ftmod = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(mod[0,:,:]))).reshape((1, npix, npix))
        chanmap = np.array([0])
        vis_degrid = degridder.degridder(uvw,
                                         ftmod,
                                         wavelength,
                                         chanmap,
                                         cell * 3600.0,
                                         (0, np.pi/4.0),
                                         (0, np.pi/4.0),
                                         kern,
                                         W,
                                         OS,
                                         "None", # no faceting
                                         "None", # no faceting
                                         "XXYY_FROM_I", 
                                         "conv_1d_axisymmetric_unpacked_gather")
        
        dec, ra = np.meshgrid(np.arange(-npix//2, npix//2) * np.deg2rad(cell),
                              np.arange(-npix//2, npix//2) * np.deg2rad(cell))
        radec = np.column_stack((ra.flatten(), dec.flatten()))
        
        vis_dft = im_to_vis(mod[0,:,:].reshape(1,1,npix*npix).T.copy(), uvw, radec, frequency)
        
        import matplotlib
        matplotlib.use("agg")  
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(vis_degrid[:,0,0].real, label="$\Re(\mathtt{degrid})$")
        plt.plot(vis_dft[:,0,0].real, label="$\Re(\mathtt{dft})$")
        plt.plot(np.abs(vis_dft[:,0,0].real - vis_degrid[:,0,0].real), label="Error")
        plt.legend()
        plt.xlabel("sample")
        plt.ylabel("Real of predicted")
        plt.savefig(os.path.join(os.environ.get("TMPDIR","/tmp"), "degrid_vs_dft_re.png"))
        plt.figure()
        plt.plot(vis_degrid[:,0,0].imag, label="$\Im(\mathtt{degrid})$")
        plt.plot(vis_dft[:,0,0].imag, label="$\Im(\mathtt{dft})$")
        plt.plot(np.abs(vis_dft[:,0,0].imag - vis_degrid[:,0,0].imag), label="Error")
        plt.legend()
        plt.xlabel("sample")
        plt.ylabel("Imag of predicted")
        plt.savefig(os.path.join(os.environ.get("TMPDIR","/tmp"), "degrid_vs_dft_im.png"))
        assert np.percentile(np.abs(vis_dft[:,0,0].real - vis_degrid[:,0,0].real),99.0) < 0.05
        assert np.percentile(np.abs(vis_dft[:,0,0].imag - vis_degrid[:,0,0].imag),99.0) < 0.05

    def test_degrid_dft_packed(self):
        # construct kernel
        W = 5
        OS = 3
        kern = kernels.pack_kernel(kernels.kbsinc(W, oversample=OS), W, oversample=OS)
        nrow = 200
        uvw = np.column_stack((5000.0 * np.cos(np.linspace(0,2*np.pi,1000)),
                               5000.0 * np.sin(np.linspace(0,2*np.pi,1000)),
                               np.zeros(1000)))
        
        pxacrossbeam = 10
        frequency = np.array([1.4e9])
        wavelength = np.array([299792458.0/f for f in frequency])

        cell = np.rad2deg(wavelength[0]/(2*max(np.max(np.abs(uvw[:,0])), 
                                               np.max(np.abs(uvw[:,1])))*pxacrossbeam))
        npix = 512
        mod = np.zeros((1, npix, npix), dtype=np.complex64)
        mod[0, npix//2 - 5, npix//2 -5] = 1.0

        ftmod = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(mod[0,:,:]))).reshape((1, npix, npix))
        chanmap = np.array([0])
        vis_degrid = degridder.degridder(uvw,
                                         ftmod,
                                         wavelength,
                                         chanmap,
                                         cell * 3600.0,
                                         (0, np.pi/4.0),
                                         (0, np.pi/4.0),
                                         kern,
                                         W,
                                         OS,
                                         "None", # no faceting
                                         "None", # no faceting
                                         "XXYY_FROM_I", 
                                         "conv_1d_axisymmetric_packed_gather")
        
        dec, ra = np.meshgrid(np.arange(-npix//2, npix//2) * np.deg2rad(cell),
                              np.arange(-npix//2, npix//2) * np.deg2rad(cell))
        radec = np.column_stack((ra.flatten(), dec.flatten()))
        
        vis_dft = im_to_vis(mod[0,:,:].reshape(1,1,npix*npix).T.copy(), uvw, radec, frequency)
        
        import matplotlib
        matplotlib.use("agg")  
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(vis_degrid[:,0,0].real, label="$\Re(\mathtt{degrid})$")
        plt.plot(vis_dft[:,0,0].real, label="$\Re(\mathtt{dft})$")
        plt.plot(np.abs(vis_dft[:,0,0].real - vis_degrid[:,0,0].real), label="Error")
        plt.legend()
        plt.xlabel("sample")
        plt.ylabel("Real of predicted")
        plt.savefig(os.path.join(os.environ.get("TMPDIR","/tmp"), "degrid_vs_dft_re_packed.png"))
        plt.figure()
        plt.plot(vis_degrid[:,0,0].imag, label="$\Im(\mathtt{degrid})$")
        plt.plot(vis_dft[:,0,0].imag, label="$\Im(\mathtt{dft})$")
        plt.plot(np.abs(vis_dft[:,0,0].imag - vis_degrid[:,0,0].imag), label="Error")
        plt.legend()
        plt.xlabel("sample")
        plt.ylabel("Imag of predicted")
        plt.savefig(os.path.join(os.environ.get("TMPDIR","/tmp"), "degrid_vs_dft_im_packed.png"))
        assert np.percentile(np.abs(vis_dft[:,0,0].real - vis_degrid[:,0,0].real),99.0) < 0.05
        assert np.percentile(np.abs(vis_dft[:,0,0].imag - vis_degrid[:,0,0].imag),99.0) < 0.05

    def test_detaper(self):
        W = 5
        OS = 3
        K = np.outer(kernels.kbsinc(W, oversample=OS), 
                     kernels.kbsinc(W, oversample=OS))
        detaper = kernels.compute_detaper(128, K, W, OS)
        detaperdft = kernels.compute_detaper_dft(128, K, W, OS)
        import matplotlib
        matplotlib.use("agg") 
        from matplotlib import pyplot as plt
        plt.figure()
        plt.subplot(131)
        plt.title("FFT detaper")
        plt.imshow(detaper)
        plt.colorbar()
        plt.subplot(132)
        plt.title("DFT detaper")
        plt.imshow(detaperdft)
        plt.colorbar()
        plt.subplot(133)
        plt.title("ABS error")
        plt.imshow(np.abs(detaper - detaperdft))
        plt.colorbar()
        plt.savefig(os.path.join(os.environ.get("TMPDIR","/tmp"), "detaper.png"))
        assert(np.percentile(np.abs(detaper - detaperdft), 99.0) < 1.0e-14)

    def test_grid_dft(self):
        # construct kernel
        W = 7
        OS = 9
        #kern = kernels.pack_kernel(kernels.kbsinc(W, oversample=OS), W, oversample=OS)
        kern = kernels.kbsinc(W, oversample=OS)
        nrow = 5000
        np.random.seed(0) 
        uvw = np.random.normal(scale=6000, size=(nrow, 3))
        uvw[:,2] = 0.0 # ignore widefield effects for now

        pxacrossbeam = 10
        frequency = np.array([30.0e9])
        wavelength = np.array([299792458.0/f for f in frequency])

        cell = np.rad2deg(wavelength[0]/(2*max(np.max(np.abs(uvw[:,0])), 
                                               np.max(np.abs(uvw[:,1])))*pxacrossbeam))
        npix = 256
        fftpad=1.25
        mod = np.zeros((1, npix, npix), dtype=np.complex64)
        for n in [int(n) for n in np.linspace(npix//8, 2*npix//5, 5)]:
           mod[0, npix//2 + n, npix//2 + n] = 1.0
           mod[0, npix//2 + n, npix//2 - n] = 1.0
           mod[0, npix//2 - n, npix//2 - n] = 1.0
           mod[0, npix//2 - n, npix//2 + n] = 1.0
           mod[0, npix//2, npix//2 + n] = 1.0
           mod[0, npix//2, npix//2 - n] = 1.0
           mod[0, npix//2 - n, npix//2] = 1.0
           mod[0, npix//2 + n, npix//2] = 1.0
        
        dec, ra = np.meshgrid(np.arange(-npix//2, npix//2) * np.deg2rad(cell),
                              np.arange(-npix//2, npix//2) * np.deg2rad(cell))
        radec = np.column_stack((ra.flatten(), dec.flatten()))
        
        vis_dft = im_to_vis(mod[0,:,:].reshape(1,1,npix*npix).T.copy(), uvw, radec, frequency).repeat(2).reshape(nrow,1,2)   
        chanmap = np.array([0])
       
        #detaper = kernels.compute_detaper(npix, np.outer(kernels.unpack_kernel(kern, W, OS), 
        #                                                 kernels.unpack_kernel(kern, W, OS)), W, OS)
        detaper = kernels.compute_detaper(int(npix*fftpad), np.outer(kern, kern), W, OS)
        vis_grid = gridder.gridder(uvw,
                                   vis_dft,
                                   wavelength,
                                   chanmap,
                                   int(npix*fftpad),
                                   cell * 3600.0,
                                   (0, np.pi/4.0),
                                   (0, np.pi/4.0),
                                   kern,
                                   W,
                                   OS,
                                   "None", # no faceting
                                   "None", # no faceting
                                   "I_FROM_XXYY",
                                   "conv_1d_axisymmetric_unpacked_scatter",
                                   do_normalize=True)
                                   
        ftvis = (np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(vis_grid[0,:,:]))).reshape((1, int(npix*fftpad), int(npix*fftpad)))).real / detaper * int(npix*fftpad) ** 2
        ftvis = ftvis[:,
                      int(npix*fftpad)//2-npix//2:int(npix*fftpad)//2-npix//2+npix,
                      int(npix*fftpad)//2-npix//2:int(npix*fftpad)//2-npix//2+npix] 
        dftvis = vis_to_im(vis_dft, uvw, radec, frequency, np.zeros(vis_dft.shape, dtype=np.bool)).T.copy().reshape(2,1,npix,npix) / nrow
        import matplotlib
        matplotlib.use("agg")  
        from matplotlib import pyplot as plt
        plt.figure()
        plt.subplot(131)
        plt.title("FFT")
        plt.imshow(ftvis[0,:,:])
        plt.colorbar()
        plt.subplot(132)
        plt.title("DFT")
        plt.imshow(dftvis[0,0,:,:])
        plt.colorbar()
        plt.subplot(133)
        plt.title("ABS diff")
        plt.imshow(np.abs(ftvis[0,:,:] - dftvis[0,0,:,:]))
        plt.colorbar()
        plt.savefig(os.path.join(os.environ.get("TMPDIR","/tmp"), "grid_diff_dft.png"))
        assert(np.percentile(np.abs(ftvis[0,:,:] - dftvis[0,0,:,:]), 95.0) < 0.15)

    def test_grid_dft_packed(self):
        # construct kernel
        W = 7
        OS = 9
        #kern = kernels.pack_kernel(kernels.kbsinc(W, oversample=OS), W, oversample=OS)
        kern = kernels.pack_kernel(kernels.kbsinc(W, oversample=OS), W, OS)
        nrow = 5000
        np.random.seed(0) 
        uvw = np.random.normal(scale=6000, size=(nrow, 3))
        uvw[:,2] = 0.0 # ignore widefield effects for now

        pxacrossbeam = 10
        frequency = np.array([30.0e9])
        wavelength = np.array([299792458.0/f for f in frequency])

        cell = np.rad2deg(wavelength[0]/(2*max(np.max(np.abs(uvw[:,0])), 
                                               np.max(np.abs(uvw[:,1])))*pxacrossbeam))
        npix = 256
        fftpad=1.25
        mod = np.zeros((1, npix, npix), dtype=np.complex64)
        for n in [int(n) for n in np.linspace(npix//8, 2*npix//5, 5)]:
           mod[0, npix//2 + n, npix//2 + n] = 1.0
           mod[0, npix//2 + n, npix//2 - n] = 1.0
           mod[0, npix//2 - n, npix//2 - n] = 1.0
           mod[0, npix//2 - n, npix//2 + n] = 1.0
           mod[0, npix//2, npix//2 + n] = 1.0
           mod[0, npix//2, npix//2 - n] = 1.0
           mod[0, npix//2 - n, npix//2] = 1.0
           mod[0, npix//2 + n, npix//2] = 1.0
        
        dec, ra = np.meshgrid(np.arange(-npix//2, npix//2) * np.deg2rad(cell),
                              np.arange(-npix//2, npix//2) * np.deg2rad(cell))
        radec = np.column_stack((ra.flatten(), dec.flatten()))
        
        vis_dft = im_to_vis(mod[0,:,:].reshape(1,1,npix*npix).T.copy(), uvw, radec, frequency).repeat(2).reshape(nrow,1,2)   
        chanmap = np.array([0])
       
        detaper = kernels.compute_detaper(int(npix*fftpad), np.outer(kernels.unpack_kernel(kern, W, OS), 
                                                                     kernels.unpack_kernel(kern, W, OS)), W, OS)
        vis_grid = gridder.gridder(uvw,
                                   vis_dft,
                                   wavelength,
                                   chanmap,
                                   int(npix*fftpad),
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
                                   
        ftvis = (np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(vis_grid[0,:,:]))).reshape((1, int(npix*fftpad), int(npix*fftpad)))).real / detaper * int(npix*fftpad) ** 2
        ftvis = ftvis[:,
                      int(npix*fftpad)//2-npix//2:int(npix*fftpad)//2-npix//2+npix,
                      int(npix*fftpad)//2-npix//2:int(npix*fftpad)//2-npix//2+npix] 
        dftvis = vis_to_im(vis_dft, uvw, radec, frequency, np.zeros(vis_dft.shape, dtype=np.bool)).T.copy().reshape(2,1,npix,npix) / nrow
        import matplotlib
        matplotlib.use("agg")  
        from matplotlib import pyplot as plt
        plt.figure()
        plt.subplot(131)
        plt.title("FFT")
        plt.imshow(ftvis[0,:,:])
        plt.colorbar()
        plt.subplot(132)
        plt.title("DFT")
        plt.imshow(dftvis[0,0,:,:])
        plt.colorbar()
        plt.subplot(133)
        plt.title("ABS diff")
        plt.imshow(np.abs(ftvis[0,:,:] - dftvis[0,0,:,:]))
        plt.colorbar()
        plt.savefig(os.path.join(os.environ.get("TMPDIR","/tmp"), "grid_diff_dft_packed.png"))
        assert(np.percentile(np.abs(ftvis[0,:,:] - dftvis[0,0,:,:]), 95.0) < 0.15) 

    # def test_adjoint_ops(self):
    #     # test adjointness of gridding and degridding operators
    #     # construct kernel
    #     W = 5
    #     OS = 3
    #     kern = kernels.pack_kernel(kernels.kbsinc(W, oversample=OS), W, oversample=OS)
    #     nrow = 2
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

    #     ftmod = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(mod[0,:,:]))).reshape((1, npix, npix))
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
    #     np.random.seed(0)
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
    #                                "conv_1d_axisymmetric_packed_scatter",
    #                                do_normalize=True)

    #     # test y* . vis_degrid(img) == grid(y) . ft(img) 
    #     lhs = np.dot(y.flatten(), vis_degrid.flatten()).real
    #     rhs = np.dot(vis_grid.flatten(), ftmod.flatten()).real
    #     import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    unittest.main()