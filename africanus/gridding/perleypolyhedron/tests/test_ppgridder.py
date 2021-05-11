# -*- coding: utf-8 -*-

import os

import numpy as np
import pytest

from africanus.gridding.perleypolyhedron import (kernels,
                                                 gridder,
                                                 degridder)
from africanus.dft.kernels import im_to_vis, vis_to_im
from africanus.coordinates import radec_to_lmn
from africanus.constants import c as lightspeed


def test_construct_kernels(tmp_path_factory):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("agg")
    plt = pytest.importorskip("matplotlib.pyplot")

    plt.figure()
    WIDTH = 5
    OVERSAMP = 101
    ll = kernels.uspace(WIDTH, OVERSAMP)
    sel = ll <= (WIDTH + 2) // 2
    plt.axvline(0.5, 0, 1, ls="--", c="k")
    plt.axvline(-0.5, 0, 1, ls="--", c="k")
    plt.plot(
        ll[sel] * OVERSAMP / 2 / np.pi,
        10 * np.log10(
            np.abs(
                np.fft.fftshift(
                    np.fft.fft(
                        kernels.kbsinc(WIDTH, oversample=OVERSAMP,
                                       order=0)[sel])))),
        label="kbsinc order 0")
    plt.plot(
        ll[sel] * OVERSAMP / 2 / np.pi,
        10 * np.log10(
            np.abs(
                np.fft.fftshift(
                    np.fft.fft(
                        kernels.kbsinc(
                            WIDTH, oversample=OVERSAMP, order=15)[sel])))),
        label="kbsinc order 15")
    plt.plot(ll[sel] * OVERSAMP / 2 / np.pi,
             10 * np.log10(
                    np.abs(
                        np.fft.fftshift(
                            np.fft.fft(
                                kernels.hanningsinc(
                                    WIDTH, oversample=OVERSAMP)[sel])))),
             label="hanning sinc")
    plt.plot(
        ll[sel] * OVERSAMP / 2 / np.pi,
        10 * np.log10(
            np.abs(
                np.fft.fftshift(
                    np.fft.fft(
                        kernels.sinc(WIDTH, oversample=OVERSAMP)[sel])))),
        label="sinc")
    plt.xlim(-10, 10)
    plt.legend()
    plt.ylabel("Response [dB]")
    plt.xlabel("FoV")
    plt.grid(True)
    plt.savefig(tmp_path_factory.mktemp("plots") / "aakernels.png")


def test_taps():
    oversample = 14
    W = 5
    taps = kernels.uspace(W, oversample=oversample)
    assert taps[oversample * ((W + 2) // 2)] == 0
    assert taps[0] == -((W + 2) // 2)
    assert taps[-oversample] == (W + 2) // 2


def test_packunpack():
    oversample = 4
    W = 3
    K = kernels.uspace(W, oversample=oversample)
    Kp = kernels.pack_kernel(K, W, oversample=oversample)
    Kup = kernels.unpack_kernel(Kp, W, oversample=oversample)
    assert np.all(K == Kup)
    assert np.allclose(K, [
        -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5,
        0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75
    ])
    assert np.allclose(Kp, [
        -2.0, -1.0, 0, 1.0, 2.0, -1.75, -0.75, 0.25, 1.25, 2.25, -1.5,
        -0.5, 0.5, 1.5, 2.5, -1.25, -0.25, 0.75, 1.75, 2.75
    ])


def test_facetcodepath():
    # construct kernel
    W = 5
    OS = 3
    kern = kernels.pack_kernel(kernels.kbsinc(W, oversample=OS),
                               W,
                               oversample=OS)

    # offset 0
    uvw = np.array([[0, 0, 0]])
    vis = np.array([[[1.0 + 0j, 1.0 + 0j]]])
    gridder.gridder(uvw, vis, np.array([1.0]), np.array([0]), 64,
                    30, (0, 0), (0, 0), kern, W, OS, "rotate",
                    "phase_rotate", "I_FROM_XXYY",
                    "conv_1d_axisymmetric_packed_scatter")


def test_degrid_dft(tmp_path_factory):
    # construct kernel
    W = 5
    OS = 3
    kern = kernels.kbsinc(W, oversample=OS)
    uvw = np.column_stack(
        (5000.0 * np.cos(np.linspace(0, 2 * np.pi, 1000)),
            5000.0 * np.sin(np.linspace(0, 2 * np.pi, 1000)), np.zeros(1000)))

    pxacrossbeam = 10
    frequency = np.array([1.4e9])
    wavelength = lightspeed / frequency

    cell = np.rad2deg(
        wavelength[0] /
        (2 * max(np.max(np.abs(uvw[:, 0])), np.max(np.abs(uvw[:, 1]))) *
            pxacrossbeam))
    npix = 512
    mod = np.zeros((1, npix, npix), dtype=np.complex64)
    mod[0, npix // 2 - 5, npix // 2 - 5] = 1.0

    ftmod = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(
        mod[0, :, :]))).reshape((1, npix, npix))
    chanmap = np.array([0])
    vis_degrid = degridder.degridder(
        uvw,
        ftmod,
        wavelength,
        chanmap,
        cell * 3600.0,
        (0, np.pi / 4.0),
        (0, np.pi / 4.0),
        kern,
        W,
        OS,
        "None",  # no faceting
        "None",  # no faceting
        "XXYY_FROM_I",
        "conv_1d_axisymmetric_unpacked_gather")

    dec, ra = np.meshgrid(
        np.arange(-npix // 2, npix // 2) * np.deg2rad(cell),
        np.arange(-npix // 2, npix // 2) * np.deg2rad(cell))
    radec = np.column_stack((ra.flatten(), dec.flatten()))

    vis_dft = im_to_vis(mod[0, :, :].reshape(1, 1, npix * npix).T.copy(),
                        uvw, radec, frequency)

    try:
        import matplotlib
    except ImportError:
        pass
    else:
        matplotlib.use("agg")
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(vis_degrid[:, 0, 0].real, label=r"$\Re(\mathtt{degrid})$")
        plt.plot(vis_dft[:, 0, 0].real, label=r"$\Re(\mathtt{dft})$")
        plt.plot(np.abs(vis_dft[:, 0, 0].real - vis_degrid[:, 0, 0].real),
                 label="Error")
        plt.legend()
        plt.xlabel("sample")
        plt.ylabel("Real of predicted")
        plt.savefig(
            os.path.join(os.environ.get("TMPDIR", "/tmp"),
                         "degrid_vs_dft_re.png"))
        plt.figure()
        plt.plot(vis_degrid[:, 0, 0].imag, label=r"$\Im(\mathtt{degrid})$")
        plt.plot(vis_dft[:, 0, 0].imag, label=r"$\Im(\mathtt{dft})$")
        plt.plot(np.abs(vis_dft[:, 0, 0].imag - vis_degrid[:, 0, 0].imag),
                 label="Error")
        plt.legend()
        plt.xlabel("sample")
        plt.ylabel("Imag of predicted")
        plt.savefig(tmp_path_factory.mktemp("degrid_dft") /
                    "degrid_vs_dft_im.png")

    assert np.percentile(
        np.abs(vis_dft[:, 0, 0].real - vis_degrid[:, 0, 0].real),
        99.0) < 0.05
    assert np.percentile(
        np.abs(vis_dft[:, 0, 0].imag - vis_degrid[:, 0, 0].imag),
        99.0) < 0.05


def test_degrid_dft_packed(tmp_path_factory):
    # construct kernel
    W = 5
    OS = 3
    kern = kernels.pack_kernel(kernels.kbsinc(W, oversample=OS),
                               W,
                               oversample=OS)
    uvw = np.column_stack(
        (5000.0 * np.cos(np.linspace(0, 2 * np.pi, 1000)),
            5000.0 * np.sin(np.linspace(0, 2 * np.pi, 1000)), np.zeros(1000)))

    pxacrossbeam = 10
    frequency = np.array([1.4e9])
    wavelength = lightspeed / frequency

    cell = np.rad2deg(
        wavelength[0] /
        (2 * max(np.max(np.abs(uvw[:, 0])), np.max(np.abs(uvw[:, 1]))) *
            pxacrossbeam))
    npix = 512
    mod = np.zeros((1, npix, npix), dtype=np.complex64)
    mod[0, npix // 2 - 5, npix // 2 - 5] = 1.0

    ftmod = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(
        mod[0, :, :]))).reshape((1, npix, npix))
    chanmap = np.array([0])
    vis_degrid = degridder.degridder(
        uvw,
        ftmod,
        wavelength,
        chanmap,
        cell * 3600.0,
        (0, np.pi / 4.0),
        (0, np.pi / 4.0),
        kern,
        W,
        OS,
        "None",  # no faceting
        "None",  # no faceting
        "XXYY_FROM_I",
        "conv_1d_axisymmetric_packed_gather")

    dec, ra = np.meshgrid(
        np.arange(-npix // 2, npix // 2) * np.deg2rad(cell),
        np.arange(-npix // 2, npix // 2) * np.deg2rad(cell))
    radec = np.column_stack((ra.flatten(), dec.flatten()))

    vis_dft = im_to_vis(mod[0, :, :].reshape(1, 1, npix * npix).T.copy(),
                        uvw, radec, frequency)

    try:
        import matplotlib
    except ImportError:
        pass
    else:
        matplotlib.use("agg")
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(vis_degrid[:, 0, 0].real, label=r"$\Re(\mathtt{degrid})$")
        plt.plot(vis_dft[:, 0, 0].real, label=r"$\Re(\mathtt{dft})$")
        plt.plot(np.abs(vis_dft[:, 0, 0].real - vis_degrid[:, 0, 0].real),
                 label="Error")
        plt.legend()
        plt.xlabel("sample")
        plt.ylabel("Real of predicted")
        plt.savefig(
            os.path.join(os.environ.get("TMPDIR", "/tmp"),
                         "degrid_vs_dft_re_packed.png"))
        plt.figure()
        plt.plot(vis_degrid[:, 0, 0].imag, label=r"$\Im(\mathtt{degrid})$")
        plt.plot(vis_dft[:, 0, 0].imag, label=r"$\Im(\mathtt{dft})$")
        plt.plot(np.abs(vis_dft[:, 0, 0].imag - vis_degrid[:, 0, 0].imag),
                 label="Error")
        plt.legend()
        plt.xlabel("sample")
        plt.ylabel("Imag of predicted")
        plt.savefig(tmp_path_factory.mktemp("degrid_dft_packed") /
                    "degrid_vs_dft_im_packed.png")

    assert np.percentile(
        np.abs(vis_dft[:, 0, 0].real - vis_degrid[:, 0, 0].real),
        99.0) < 0.05
    assert np.percentile(
        np.abs(vis_dft[:, 0, 0].imag - vis_degrid[:, 0, 0].imag),
        99.0) < 0.05


def test_detaper(tmp_path_factory):
    W = 5
    OS = 3
    K1D = kernels.kbsinc(W, oversample=OS)
    K2D = np.outer(K1D, K1D)
    detaper = kernels.compute_detaper(128, K2D, W, OS)
    detaperdft = kernels.compute_detaper_dft(128, K2D, W, OS)
    detaperdftsep = kernels.compute_detaper_dft_seperable(128, K1D, W, OS)

    try:
        import matplotlib
    except ImportError:
        pass
    else:
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
        plt.savefig(tmp_path_factory.mktemp("detaper") / "detaper.png")

    assert (np.percentile(np.abs(detaper - detaperdft), 99.0) < 1.0e-14)
    assert (np.max(np.abs(detaperdft - detaperdftsep)) < 1.0e-14)


def test_grid_dft(tmp_path_factory):
    # construct kernel
    W = 7
    OS = 9
    kern = kernels.kbsinc(W, oversample=OS)
    nrow = 5000
    np.random.seed(0)
    uvw = np.random.normal(scale=6000, size=(nrow, 3))
    uvw[:, 2] = 0.0  # ignore widefield effects for now

    pxacrossbeam = 10
    frequency = np.array([30.0e9])
    wavelength = lightspeed / frequency

    cell = np.rad2deg(
        wavelength[0] /
        (2 * max(np.max(np.abs(uvw[:, 0])), np.max(np.abs(uvw[:, 1]))) *
            pxacrossbeam))
    npix = 256
    fftpad = 1.25
    mod = np.zeros((1, npix, npix), dtype=np.complex64)
    for n in [int(n) for n in np.linspace(npix // 8, 2 * npix // 5, 5)]:
        mod[0, npix // 2 + n, npix // 2 + n] = 1.0
        mod[0, npix // 2 + n, npix // 2 - n] = 1.0
        mod[0, npix // 2 - n, npix // 2 - n] = 1.0
        mod[0, npix // 2 - n, npix // 2 + n] = 1.0
        mod[0, npix // 2, npix // 2 + n] = 1.0
        mod[0, npix // 2, npix // 2 - n] = 1.0
        mod[0, npix // 2 - n, npix // 2] = 1.0
        mod[0, npix // 2 + n, npix // 2] = 1.0

    dec, ra = np.meshgrid(
        np.arange(-npix // 2, npix // 2) * np.deg2rad(cell),
        np.arange(-npix // 2, npix // 2) * np.deg2rad(cell))
    radec = np.column_stack((ra.flatten(), dec.flatten()))

    vis_dft = im_to_vis(mod[0, :, :].reshape(1, 1,
                                             npix * npix).T.copy(), uvw,
                        radec, frequency).repeat(2).reshape(nrow, 1, 2)
    chanmap = np.array([0])

    detaper = kernels.compute_detaper(int(npix * fftpad),
                                      np.outer(kern, kern), W, OS)
    vis_grid = gridder.gridder(
        uvw,
        vis_dft,
        wavelength,
        chanmap,
        int(npix * fftpad),
        cell * 3600.0,
        (0, np.pi / 4.0),
        (0, np.pi / 4.0),
        kern,
        W,
        OS,
        "None",  # no faceting
        "None",  # no faceting
        "I_FROM_XXYY",
        "conv_1d_axisymmetric_unpacked_scatter",
        do_normalize=True)

    ftvis = (np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(vis_grid[0, :, :]))).reshape(
            (1, int(npix * fftpad), int(
                npix * fftpad)))).real / detaper * int(npix * fftpad)**2
    ftvis = ftvis[:,
                  int(npix * fftpad) // 2 -
                  npix // 2:int(npix * fftpad) // 2 - npix // 2 + npix,
                  int(npix * fftpad) // 2 -
                  npix // 2:int(npix * fftpad) // 2 - npix // 2 + npix]
    dftvis = vis_to_im(vis_dft, uvw, radec, frequency,
                       np.zeros(vis_dft.shape,
                                dtype=np.bool_)).T.copy().reshape(
                                    2, 1, npix, npix) / nrow

    try:
        import matplotlib
    except ImportError:
        pass
    else:
        matplotlib.use("agg")
        from matplotlib import pyplot as plt
        plt.figure()
        plt.subplot(131)
        plt.title("FFT")
        plt.imshow(ftvis[0, :, :])
        plt.colorbar()
        plt.subplot(132)
        plt.title("DFT")
        plt.imshow(dftvis[0, 0, :, :])
        plt.colorbar()
        plt.subplot(133)
        plt.title("ABS diff")
        plt.imshow(np.abs(ftvis[0, :, :] - dftvis[0, 0, :, :]))
        plt.colorbar()
        plt.savefig(tmp_path_factory.mktemp("grid_dft") /
                    "grid_diff_dft.png")

    assert (np.percentile(np.abs(ftvis[0, :, :] - dftvis[0, 0, :, :]),
                          95.0) < 0.15)


def test_grid_dft_packed(tmp_path_factory):
    # construct kernel
    W = 7
    OS = 1009
    kern = kernels.pack_kernel(kernels.kbsinc(W, oversample=OS), W, OS)
    nrow = 5000
    np.random.seed(0)
    uvw = np.random.normal(scale=6000, size=(nrow, 3))
    uvw[:, 2] = 0.0  # ignore widefield effects for now

    pxacrossbeam = 10
    frequency = np.array([30.0e9])
    wavelength = lightspeed / frequency

    cell = np.rad2deg(
        wavelength[0] /
        (2 * max(np.max(np.abs(uvw[:, 0])), np.max(np.abs(uvw[:, 1]))) *
            pxacrossbeam))
    npix = 256
    fftpad = 1.25
    mod = np.zeros((1, npix, npix), dtype=np.complex64)
    for n in [int(n) for n in np.linspace(npix // 8, 2 * npix // 5, 5)]:
        mod[0, npix // 2 + n, npix // 2 + n] = 1.0
        mod[0, npix // 2 + n, npix // 2 - n] = 1.0
        mod[0, npix // 2 - n, npix // 2 - n] = 1.0
        mod[0, npix // 2 - n, npix // 2 + n] = 1.0
        mod[0, npix // 2, npix // 2 + n] = 1.0
        mod[0, npix // 2, npix // 2 - n] = 1.0
        mod[0, npix // 2 - n, npix // 2] = 1.0
        mod[0, npix // 2 + n, npix // 2] = 1.0

    dec, ra = np.meshgrid(
        np.arange(-npix // 2, npix // 2) * np.deg2rad(cell),
        np.arange(-npix // 2, npix // 2) * np.deg2rad(cell))
    radec = np.column_stack((ra.flatten(), dec.flatten()))

    vis_dft = im_to_vis(mod[0, :, :].reshape(1, 1,
                                             npix * npix).T.copy(), uvw,
                        radec, frequency).repeat(2).reshape(nrow, 1, 2)
    chanmap = np.array([0])
    detaper = kernels.compute_detaper_dft_seperable(
        int(npix * fftpad), kernels.unpack_kernel(kern, W, OS), W, OS)
    vis_grid = gridder.gridder(
        uvw,
        vis_dft,
        wavelength,
        chanmap,
        int(npix * fftpad),
        cell * 3600.0,
        (0, np.pi / 4.0),
        (0, np.pi / 4.0),
        kern,
        W,
        OS,
        "None",  # no faceting
        "None",  # no faceting
        "I_FROM_XXYY",
        "conv_1d_axisymmetric_packed_scatter",
        do_normalize=True)

    ftvis = (np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(vis_grid[0, :, :]))).reshape(
            (1, int(npix * fftpad), int(
                npix * fftpad)))).real / detaper * int(npix * fftpad)**2
    ftvis = ftvis[:,
                  int(npix * fftpad) // 2 -
                  npix // 2:int(npix * fftpad) // 2 - npix // 2 + npix,
                  int(npix * fftpad) // 2 -
                  npix // 2:int(npix * fftpad) // 2 - npix // 2 + npix]
    dftvis = vis_to_im(vis_dft, uvw, radec, frequency,
                       np.zeros(vis_dft.shape,
                                dtype=np.bool_)).T.copy().reshape(
                                    2, 1, npix, npix) / nrow

    try:
        import matplotlib
    except ImportError:
        pass
    else:
        matplotlib.use("agg")
        from matplotlib import pyplot as plt
        plt.figure()
        plt.subplot(131)
        plt.title("FFT")
        plt.imshow(ftvis[0, :, :])
        plt.colorbar()
        plt.subplot(132)
        plt.title("DFT")
        plt.imshow(dftvis[0, 0, :, :])
        plt.colorbar()
        plt.subplot(133)
        plt.title("ABS diff")
        plt.imshow(np.abs(ftvis[0, :, :] - dftvis[0, 0, :, :]))
        plt.colorbar()
        plt.savefig(tmp_path_factory.mktemp("grid_dft_packed") /
                    "grid_diff_dft_packed.png")

    assert (np.percentile(np.abs(ftvis[0, :, :] - dftvis[0, 0, :, :]),
                          95.0) < 0.15)


def test_wcorrection_faceting_backward(tmp_path_factory):
    # construct kernel
    W = 5
    OS = 9
    kern = kernels.pack_kernel(kernels.kbsinc(W, oversample=OS), W, OS)
    nrow = 5000
    np.random.seed(0)
    # simulate some ficticious baselines rotated by an hour angle
    uvw = np.zeros((nrow, 3), dtype=np.float64)
    blpos = np.random.uniform(26, 10000, size=(25, 3))
    ntime = int(nrow / 25.0)
    d0 = np.pi / 4.0
    for n in range(25):
        for ih0, h0 in enumerate(
                np.linspace(np.deg2rad(-20), np.deg2rad(20), ntime)):
            s = np.sin
            c = np.cos
            R = np.array([[s(h0), c(h0), 0],
                          [-s(d0) * c(h0),
                           s(d0) * s(h0),
                           c(d0)], [c(d0) * c(h0), -c(d0) * s(h0),
                                    s(d0)]])
            uvw[n * ntime + ih0, :] = np.dot(R, blpos[n, :].T)

    pxacrossbeam = 5
    frequency = np.array([1.4e9])
    wavelength = lightspeed / frequency

    cell = np.rad2deg(
        wavelength[0] /
        (max(np.max(np.abs(uvw[:, 0])), np.max(np.abs(uvw[:, 1]))) *
            pxacrossbeam))
    npix = 2048
    npixfacet = 100
    fftpad = 1.1
    mod = np.ones((1, 1, 1), dtype=np.complex64)
    deltaradec = np.array(
        [[600 * np.deg2rad(cell), 600 * np.deg2rad(cell)]])
    lm = radec_to_lmn(deltaradec + np.array([[0, d0]]),
                      phase_centre=np.array([0, d0]))

    vis_dft = im_to_vis(mod, uvw, lm[:, 0:2],
                        frequency).repeat(2).reshape(nrow, 1, 2)
    chanmap = np.array([0])

    detaper = kernels.compute_detaper_dft_seperable(
        int(npix * fftpad), kernels.unpack_kernel(kern, W, OS), W, OS)
    vis_grid_nofacet = gridder.gridder(
        uvw,
        vis_dft,
        wavelength,
        chanmap,
        int(npix * fftpad),
        cell * 3600.0,
        (0, d0),
        (0, d0),
        kern,
        W,
        OS,
        "None",  # no faceting
        "None",  # no faceting
        "I_FROM_XXYY",
        "conv_1d_axisymmetric_packed_scatter",
        do_normalize=True)
    ftvis = (np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(vis_grid_nofacet[0, :, :]))).reshape(
            (1, int(npix * fftpad), int(
                npix * fftpad)))).real / detaper * int(npix * fftpad)**2
    ftvis = ftvis[:,
                  int(npix * fftpad) // 2 -
                  npix // 2:int(npix * fftpad) // 2 - npix // 2 + npix,
                  int(npix * fftpad) // 2 -
                  npix // 2:int(npix * fftpad) // 2 - npix // 2 + npix]

    detaper_facet = kernels.compute_detaper_dft_seperable(
        int(npixfacet * fftpad), kernels.unpack_kernel(kern, W, OS), W, OS)
    vis_grid_facet = gridder.gridder(
        uvw,
        vis_dft,
        wavelength,
        chanmap,
        int(npixfacet * fftpad),
        cell * 3600.0, (deltaradec + np.array([[0, d0]]))[0, :], (0, d0),
        kern,
        W,
        OS,
        "rotate",
        "phase_rotate",
        "I_FROM_XXYY",
        "conv_1d_axisymmetric_packed_scatter",
        do_normalize=True)
    ftvisfacet = (np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(vis_grid_facet[0, :, :]))).reshape(
            (1, int(npixfacet * fftpad), int(
                npixfacet * fftpad)))).real / detaper_facet * int(
                    npixfacet * fftpad)**2
    ftvisfacet = ftvisfacet[:,
                            int(npixfacet * fftpad) // 2 -
                            npixfacet // 2:int(npixfacet * fftpad) // 2 -
                            npixfacet // 2 + npixfacet,
                            int(npixfacet * fftpad) // 2 -
                            npixfacet // 2:int(npixfacet * fftpad) // 2 -
                            npixfacet // 2 + npixfacet]

    try:
        import matplotlib
    except ImportError:
        pass
    else:
        matplotlib.use("agg")
        from matplotlib import pyplot as plt
        plot_dir = tmp_path_factory.mktemp("wcorrection_backward")

        plt.figure()
        plt.subplot(121)
        plt.imshow(ftvis[0, 1624 - 50:1624 + 50, 1447 - 50:1447 + 50])
        plt.colorbar()
        plt.title("Offset FFT (peak={0:.1f})".format(np.max(ftvis)))
        plt.subplot(122)
        plt.imshow(ftvisfacet[0, :, :])
        plt.colorbar()
        plt.title("Faceted FFT (peak={0:.1f})".format(np.max(ftvisfacet)))
        plt.savefig(plot_dir / "facet_imaging.png")

    assert (np.abs(np.max(ftvisfacet[0, :, :]) - 1.0) < 1.0e-6)


def test_wcorrection_faceting_forward(tmp_path_factory):
    # construct kernel
    W = 5
    OS = 9
    kern = kernels.pack_kernel(kernels.kbsinc(W, oversample=OS), W, OS)
    nrow = 5000
    np.random.seed(0)
    # simulate some ficticious baselines rotated by an hour angle
    uvw = np.zeros((nrow, 3), dtype=np.float64)
    blpos = np.random.uniform(26, 10000, size=(25, 3))
    ntime = int(nrow / 25.0)
    d0 = np.pi / 4.0
    for n in range(25):
        for ih0, h0 in enumerate(
                np.linspace(np.deg2rad(-20), np.deg2rad(20), ntime)):
            s = np.sin
            c = np.cos
            R = np.array([[s(h0), c(h0), 0],
                          [-s(d0) * c(h0),
                           s(d0) * s(h0),
                           c(d0)], [c(d0) * c(h0), -c(d0) * s(h0),
                                    s(d0)]])
            uvw[n * ntime + ih0, :] = np.dot(R, blpos[n, :].T)

    pxacrossbeam = 5
    frequency = np.array([1.4e9])
    wavelength = lightspeed / frequency

    cell = np.rad2deg(
        wavelength[0] /
        (max(np.max(np.abs(uvw[:, 0])), np.max(np.abs(uvw[:, 1]))) *
            pxacrossbeam))
    npixfacet = 100
    mod = np.ones((1, 1, 1), dtype=np.complex64)
    deltaradec = np.array([[20 * np.deg2rad(cell), 20 * np.deg2rad(cell)]])
    lm = radec_to_lmn(deltaradec + np.array([[0, d0]]),
                      phase_centre=np.array([0, d0]))

    vis_dft = im_to_vis(mod, uvw, lm[:, 0:2],
                        frequency).repeat(2).reshape(nrow, 1, 2)
    chanmap = np.array([0])
    ftmod = np.ones((1, npixfacet, npixfacet),
                    dtype=np.complex64)  # point source at centre of facet
    vis_degrid = degridder.degridder(
        uvw,
        ftmod,
        wavelength,
        chanmap,
        cell * 3600.0,
        (deltaradec + np.array([[0, d0]]))[0, :],
        (0, d0),
        kern,
        W,
        OS,
        "rotate",  # no faceting
        "phase_rotate",  # no faceting
        "XXYY_FROM_I",
        "conv_1d_axisymmetric_packed_gather")

    try:
        import matplotlib
    except ImportError:
        pass
    else:
        matplotlib.use("agg")
        from matplotlib import pyplot as plt
        plot_dir = tmp_path_factory.mktemp("wcorrection_forward")

        plt.figure()
        plt.plot(vis_degrid[:, 0, 0].real,
                 label=r"$\Re(\mathtt{degrid facet})$")
        plt.plot(vis_dft[:, 0, 0].real, label=r"$\Re(\mathtt{dft})$")
        plt.plot(np.abs(vis_dft[:, 0, 0].real - vis_degrid[:, 0, 0].real),
                 label="Error")
        plt.legend()
        plt.xlabel("sample")
        plt.ylabel("Real of predicted")
        plt.savefig(plot_dir / "facet_degrid_vs_dft_re_packed.png")
        plt.figure()
        plt.plot(vis_degrid[:, 0, 0].imag,
                 label=r"$\Im(\mathtt{degrid facet})$")
        plt.plot(vis_dft[:, 0, 0].imag, label=r"$\Im(\mathtt{dft})$")
        plt.plot(np.abs(vis_dft[:, 0, 0].imag - vis_degrid[:, 0, 0].imag),
                 label="Error")
        plt.legend()
        plt.xlabel("sample")
        plt.ylabel("Imag of predicted")
        plt.savefig(plot_dir / "facet_degrid_vs_dft_im_packed.png")

    assert np.percentile(
        np.abs(vis_dft[:, 0, 0].real - vis_degrid[:, 0, 0].real),
        99.0) < 0.05
    assert np.percentile(
        np.abs(vis_dft[:, 0, 0].imag - vis_degrid[:, 0, 0].imag),
        99.0) < 0.05
