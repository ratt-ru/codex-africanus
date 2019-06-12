#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from africanus.constants import c as lightspeed
from africanus.util.numba import jit
from africanus.util.requirements import requires_optional

try:
    from scipy.interpolate import interp1d
except ImportError as e:
    opt_import_error = e
else:
    opt_import_error = None


@jit(nopython=True, nogil=True, cache=True)
def _gen_coeffs(x, y, order):
    ncols = (order + 1)**2
    coeffs = np.empty((x.size, ncols), dtype=x.dtype)
    c = 0

    for i in range(order + 1):
        for j in range(order + 1):
            for k in range(x.size):
                coeffs[k, c] = x[k]**i * y[k]**j

            c += 1

    return coeffs


def polyfit2d(x, y, z, order=3):
    """
    Given ``x`` and ``y`` data points and ``z``, some
    values related to ``x`` and ``y``, fit a polynomial
    of order ``order`` to ``z``.

    Derived from https://stackoverflow.com/a/7997925
    """
    return np.linalg.lstsq(_gen_coeffs(x, y, order), z, rcond=-1)[0]


@jit(nopython=True, nogil=True, cache=True)
def polyval2d(x, y, coeffs):
    """
    Reproduce values from a two-dimensional polynomial fit.

    Derived from https://stackoverflow.com/a/7997925
    """
    order = int(np.sqrt(coeffs.size)) - 1
    z = np.zeros_like(x)
    c = 0

    for i in range(order + 1):
        for j in range(order + 1):
            a = coeffs[c]
            for k in range(x.shape[0]):
                z[k] += a * x[k]**i * y[k]**j

            c += 1

    return z

# Magic spheroidal coefficients
# These derived from
# Rational Approximations to Selected 0-order Spheroidal Functions
# https://library.nrao.edu/public/memos/vla/comp/VLAC_156.pdf
# Table IIIA (c) and Table IIIB (c) respectively

# These values exist for a support width m = 6,
# First elements are for |nu| < 0.75 and second for 0.75 <= |nu| <= 1.0

# NOTE(sjperkins)
# The above support width is generally
# much smaller than the filter support sizes


P = np.array([
    [8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1],
    [4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2]])

Q = np.array([
    [1.0000000e0, 8.212018e-1, 2.078043e-1],
    [1.0000000e0, 9.599102e-1, 2.918724e-1]])


@jit(nopython=True, nogil=True, cache=True)
def spheroidal_2d(npix, factor=1.0):
    result = np.empty((npix, npix), dtype=np.float64)
    c = np.linspace(-1.0, 1.0, npix)

    for y, yc in enumerate(c):
        y_sqrd = yc**2
        for x, xc in enumerate(c):
            r = np.sqrt(xc**2 + y_sqrd)*factor

            if r >= 0.0 and r < 0.75:
                poly = 0
                end = 0.75
            elif r >= 0.75 and r <= 1.00:
                poly = 1
                end = 1.00
            else:
                result[y, x] = 0.0
                continue

            sP = P[poly]
            sQ = Q[poly]

            nu_sqrd = r**2
            del_nu_sqrd = nu_sqrd - end*end

            top = sP[0]
            del_nu_sqrd_pow = del_nu_sqrd

            for i in range(1, 5):
                top += sP[i]*del_nu_sqrd_pow
                del_nu_sqrd_pow *= del_nu_sqrd

            bot = sQ[0]
            del_nu_sqrd_pow = del_nu_sqrd

            for i in range(1, 3):
                bot += sQ[i]*del_nu_sqrd_pow
                del_nu_sqrd_pow *= del_nu_sqrd

            result[y, x] = (1.0 - nu_sqrd) * (top/bot)

    return result


def np_spheroidal_2d(npix, factor=1.0):
    """ Numpy implementation of spheroidal_2d """
    x = np.mgrid[-1:1:1j*npix]**2

    assert np.all(x == np.linspace(-1, 1, npix, endpoint=True)**2)

    r = np.sqrt(x[:, None] + x[None, :])*factor

    bin1 = np.logical_and(r >= 0.0, r < 0.75)
    bin2 = np.logical_and(r >= 0.75, r <= 1.00)
    bin3 = np.invert(np.logical_or(bin1, bin2))

    def _eval_spheroid(nu, part, end):
        sP = P[part]
        sQ = Q[part]

        nu_sqrd = nu**2
        del_nu_sqrd = nu_sqrd - end*end
        powers = del_nu_sqrd[:, None]**np.arange(5)

        top = np.sum(sP[None, :]*powers, axis=1)
        bot = np.sum(sQ[None, :]*powers[:, 0:3], axis=1)

        return (1.0 - nu_sqrd) * (top/bot)

    result = np.empty_like(r)

    result[bin1] = _eval_spheroid(r[bin1], 0, 0.75)
    result[bin2] = _eval_spheroid(r[bin2], 1, 1.00)
    result[bin3] = 0.0

    return result


def zero_pad(img, npix):
    """ Zero pad ``img`` up to ``npix`` """

    if isinstance(npix, int):
        npix = (npix,)*img.ndim

    padding = []

    for dim, npix_ in zip(img.shape, npix):
        # Pad and half-pad amount
        p = npix_ - dim
        hp = p // 2

        # Pad the imagew
        padding.append((hp, hp) if p % 2 == 0 else (hp+1, hp))

    return np.pad(img, padding, 'constant', constant_values=0)


def spheroidal_aa_filter(npix, support=11, spheroidal_support=111):
    # Convolution filter
    cf = spheroidal_2d(spheroidal_support).astype(np.complex128)
    # Fourier transformed convolution filter
    cf_size = cf.real.dtype.type(cf.size)
    fcf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(cf))) / cf_size

    # Cut the support out
    xc = spheroidal_support//2
    start = xc - support//2
    end = 1 + xc + support//2
    fcf = fcf[start:end, start:end].copy()

    # Inverse fourier transform of the cut
    # if_cut_fcf = np.fft.ifft2(fcf)

    # Pad and ifft2 the fourier transformed convolution filter
    zfcf = zero_pad(fcf, npix)
    zfcf_size = zfcf.real.dtype.type(zfcf.size)
    ifzfcf = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(zfcf))) * zfcf_size
    ifzfcf[ifzfcf < 0] = 1e-10

    return cf, fcf, ifzfcf


def delta_n_coefficients(l0, m0, radius=1., order=4):
    """
    Returns polynomical coefficients representing the difference
    of coordinate n between a grid of (l,m) values centred
    around (l0, m0).
    """

    p = 100

    l, m = np.mgrid[l0 - radius:l0 + radius:p*1j, m0 - radius:m0 + radius:p*1j]

    dl = l - l0
    dm = m - m0

    dl = dl.flatten()
    dm = dm.flatten()
    y = np.sqrt(1-(dl + l0)**2 - (dm + m0)**2) - np.sqrt(1 - l0**2 - m0**2)
    coeff = polyfit2d(dl, dm, y, order=order)
    C = coeff.reshape((order + 1, order + 1))
    Cl = C[0, 1]
    Cm = C[1, 0]
    C[0, 1] = 0
    C[1, 0] = 0

    return Cl, Cm, coeff


@jit(nopython=True, nogil=True, cache=True)
def reorganise_convolution_filter(cf, oversampling):
    """
    TODO(sjperkins)
    Understand what's going on here...

    Parameters
    ----------
    cf : np.ndarray
        Oversampled convolution filter
    oversampling : integer
        Oversampling factor

    Returns
    -------
    np.ndarray
        Reorganised convolution filter

    """
    support = cf.shape[0] // oversampling
    result = np.empty((oversampling, oversampling, support, support),
                      dtype=cf.dtype)

    for i in range(oversampling):
        for j in range(oversampling):
            result[i, j, :, :] = cf[i::oversampling, j::oversampling]

    return result.reshape(cf.shape)


@requires_optional("scipy.interpolate", opt_import_error)
def find_max_support(radius, maxw, min_wave):
    """
    Find the maximum support
    """
    # Assumed maximum support
    max_support = 501

    # Work out the spheroidal convolution filter for
    # the maximum support size
    _, _, spheroidal_w = spheroidal_aa_filter(max_support)

    # Compute l, m and n-1 over the area of maximum support
    ex = radius*np.sqrt(2.)
    l, m = np.mgrid[-ex:ex:max_support*1j, -ex:ex:max_support*1j]
    n_1 = np.sqrt(1.0 - l**2 - m ** 2) - 1.0

    # Compute the w term
    w = np.exp(-2.0*1j*np.pi*(maxw/min_wave)*n_1)*spheroidal_w
    w_size = w.dtype.type(w.size)
    fw = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(w))) / w_size

    # Want to interpolate across fw.
    # I think we take a slice across fw at the halfway point
    # because fw is symmetric
    fw1d = np.abs(fw[(max_support-1)//2, :])
    # normalise
    fw1d /= np.max(fw1d)
    # Then take half again due to symmetry
    fw1d = fw1d[(max_support-1)//2::]

    ind = np.argsort(fw1d)

    # TODO(sjperkins)
    # Find a less clunky way to find the maximum support
    # Why are we sampling some value right near the start?
    # Replace scipy interp with numpy.interp?
    interp_fn = interp1d(fw1d[ind], np.arange(fw1d.shape[0])[ind])
    max_support = int(interp_fn(1./1000))

    return max_support


def wplanes(nwplanes, cell_size, support, maxw,
            npix, oversampling,
            lmshift, frequencies):
    """
    Compute w projection planes and their conjugates

    Parameters
    ----------
    nwplanes : integer
        Number of W planes
    cell_size : float
        Cell size in arc seconds
    support : integer
        Support, in pixels
    maxw : float
        Maximum W coordinate value
    npix : integer
        Number of pixels
    oversampling : integer
        oversampling factor, in pixels
    lmshift (optional) : tuple
        (l0, m0) coordinate for this convolution filter.
        Default to :code:`(0.0, 0.0)` if ``None``.
    frequencies : np.ndarray
        Array of frequencies

    Returns
    -------
    cu : float
        First polynomial U coefficient
    cv : float
        First polynomial V coefficient
    w_proj_filters : list
        w projection filters
    w_proj_conj_filters : list
        conjugate w projection filters
    """

    # Radius in radians, given cell size and number of pixels
    radius = np.deg2rad((npix / 2.0)*cell_size / 3600.0)

    # Minimum wavelength relates to max frequency
    min_wave = lightspeed / frequencies.max()

    # Find the maximum support
    max_support = find_max_support(radius, maxw, min_wave)

    # W values for each plane
    w_values = np.linspace(0, maxw, nwplanes)

    # Create supports for each w plane
    w_supports = np.linspace(support, max(max_support, support),
                             nwplanes, dtype=np.int64)

    # Make any even support odd
    w_supports[w_supports % 2 == 0] += 1

    # Extract lm coordinates if given
    l0, m0 = lmshift if lmshift else (0.0, 0.0)

    # Work out general dn polynomial coefficients
    cu, cv, poly_coeffs = delta_n_coefficients(l0, m0, 3*radius, order=5)

    wplanes = []
    wplanes_conj = []

    # For each w plane and associated support
    for i, (plane_w, w_support) in enumerate(zip(w_values, w_supports)):
        # Normalise plane w
        norm_plane_w = plane_w / min_wave

        # Calculate the spheroidal for the given support
        _, _, spheroidal_pw = spheroidal_aa_filter(w_support)

        # Fit n-1 for this w plane using
        # delta n polynomial coefficients
        ex = radius - radius/w_support
        l, m = np.mgrid[-ex:ex:w_support*1j, -ex:ex:w_support*1j]
        n_1 = polyval2d(l, m, poly_coeffs)

        # Multiply in complex exponential
        # and the spheroidal for this plane
        # Convolution theorem?
        w = np.exp(-2.0*1j*np.pi*norm_plane_w*n_1)*np.abs(spheroidal_pw)

        # zero pad w, adding oversampling
        zw = zero_pad(w, w.shape[0]*oversampling)
        zw_conj = np.conj(zw)

        # Now fft2 zero padded w and it's conjugate
        zw_size = zw.dtype.type(zw.size)
        zw_conj_size = zw_conj.dtype.type(zw_conj.size)
        fzw = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(zw))) / zw_size
        fzw_conj = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(zw_conj))) / zw_conj_size

        # TODO(sjperkins)
        # Understand what is going on here...
        fzw = reorganise_convolution_filter(fzw, oversampling)
        fzw_conj = reorganise_convolution_filter(fzw_conj, oversampling)

        # Cast to complex64, ensure aligned and contiguous
        fzw = np.require(fzw.astype(np.complex64),
                         requirements=["A", "C"])
        fzw_conj = np.require(fzw_conj.astype(np.complex64),
                              requirements=["A", "C"])

        wplanes.append(fzw)
        wplanes_conj.append(fzw_conj)

    return cu, cv, wplanes, wplanes_conj


if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser()
    p.add_argument("-np", "--npix", default=129, type=int)
    p.add_argument("-s", "--support", default=11, type=int)
    p.add_argument("-ss", "--spheroidal-support", default=111, type=int)
    p.add_argument("-d", "--display", default="cf",
                   choices=["cf", "fcf", "ifzfcf"])

    args = p.parse_args()

    wplanes(nwplanes=5, cell_size=10, support=15, maxw=30000, npix=101,
            oversampling=11, lmshift=None,
            frequencies=np.linspace(.856e9, 2*.856e9, 64))

    cf, fcf, ifzfcf = spheroidal_aa_filter(args.npix,
                                           args.support,
                                           args.spheroidal_support)

    print("Convolution filter", cf.shape)
    print("Fourier transformed Convolution Filter", fcf.shape)
    print("Inverse Fourier transform of Zero-padded fcf", ifzfcf.shape)

    assert np.allclose(spheroidal_2d(args.spheroidal_support),
                       np_spheroidal_2d(args.spheroidal_support))

    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    if args.display == 'cf':
        data = cf
    elif args.display == "fcf":
        data = fcf
    elif args.display == "ifzfcf":
        data = ifzfcf
    else:
        raise ValueError("Invalid choice %s" % args.display)

    X, Y = np.mgrid[-1:1:1j*data.shape[0], -1:1:1j*data.shape[1]]

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X, Y, np.abs(data), cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
