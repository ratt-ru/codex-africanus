import numba
import numpy as np
from numpy import sqrt, exp
from africanus.constants import c as lightspeed
from africanus.constants import minus_two_pi_over_c

e = 2.7182818284590452353602874713527
square_root_of_pi = 1.77245385091

#@numba.jit(nogil=True, nopython=True, cache=True)
def hermite(n, x):
    if n==0:
        return 1
    elif n==1:
        return 2*x
    else:
        return 2*x*hermite(n-1,x)-2*(n-1)*hermite(n-2,x)


#@numba.jit(nogil=True, nopython=True, cache=True)
def factorial(n):
    if n <= 1:
        return 1
    ans = 1
    for i in range(1, n):
        ans *= i
    return ans * n

#@numba.jit(nogil=True, nopython=True, cache=True)
def basis_function(n, xx, beta, fourier=False, delta_x=None):
    if fourier:
        x = 2*np.pi*xx
        scale = 1.0/beta
    else:
        x = xx
        scale = beta
    basis_component = 1.0/np.sqrt(2.0**n * np.sqrt(np.pi) * factorial(n) * scale)
    exponential_component = hermite(n, x / scale) * np.exp(-x**2 / (2.0*scale**2))
    if fourier:
        return 1.0j**n * basis_component * exponential_component * np.sqrt(2*np.pi)/delta_x
    else:
        return basis_component * exponential_component
"""
#@numba.jit(nogil=True, nopython=True, cache=True)
def phase_steer_and_w_correct(uvw, lm_source_center, frequency):
    l0, m0 = lm_source_center
    n0 = np.sqrt(1.0-l0**2-m0**2)
    u, v, w = uvw
    real_phase = minus_two_pi_over_c * frequency * (u*l0 + v*m0 + w*(n0-1))
    return np.exp(1.0j*real_phase)
"""

#@numba.jit(nogil=True, nopython=True, cache=True)
def shapelet(coords, frequency, coeffs, beta, delta_lm, dtype=np.complex128):
    """
    shapelet: outputs visibilities corresponding to that of a shapelet
    Inputs:
        coords: coordinates in (u,v) space with shape (nrow, 3)
        frequency: frequency values with shape (nchan,)
        coeffs: shapelet coefficients with shape, where coeffs[3, 4] = coeffs_l[3] * coeffs_m[4] (nsrc, nmax1, nmax2)
        beta: characteristic shapelet size with shape (nsrc, 2)
        delta_l: pixel size in l dim
        delta_m: pixel size in m dim
        lm: source center coordinates of shape (nsource, 2)
    Returns:
        out_shapelets: Shapelet with shape (nrow, nchan, nsrc)
    """
    nrow = coords.shape[0]
    nsrc, nmax1, nmax2 = coeffs.shape
    nchan = frequency.size
    out_shapelets = np.empty((nrow, nchan, nsrc), dtype=np.complex128)
    delta_l, delta_m = delta_lm
    for row in range(nrow):
        u, v, w = coords[row, :]
        # delta_l = 1.0/np.max(coords[:, 0])
        # delta_m = 1.0/np.max(coords[:, 1])
        for chan in range(nchan):
            fu = u * 2 * np.pi * frequency[chan] / lightspeed
            fv = v * 2 * np.pi * frequency[chan] / lightspeed
            for src in range(nsrc):
                beta_u, beta_v = beta[src, :]
                tmp_shapelet = np.zeros(1, dtype=dtype)
                for n1 in range(nmax1):
                    for n2 in range(nmax2):
                        tmp_shapelet += coeffs[src, n1, n2] * basis_function(n1, fu, beta_u, True, delta_x=delta_l) \
                            * basis_function(n2, fv, beta_v, True, delta_x=delta_m)
                out_shapelets[row, chan, src] = tmp_shapelet[0]
    return out_shapelets

#@numba.jit(nogil=True, nopython=True, cache=True)
def shapelet_1d(u, coeffs, fourier, delta_x=None, beta=1.0):
    """
    The one dimensional shapelet. Default is to return the
    dimensionless version. 
    Parameters
    ----------
    u : :class:`numpy.ndarray`
        Array of coordinates at which to evaluate the shapelet
        of shape (nrow)
    coeffs : :class:`numpy.ndarray`
        Array of shapelet coefficients of chape (ncoeff)
    fourier : bool
        Whether to evaluate the shapelet in Fourier space
        or in signal space
    beta : float, optional
        The scale parameter for the shapelet. If fourier is
        true the scale is 1/beta
    Returns
    -------
    out : :class:`numpy.ndarray`
        The shapelet evaluated at u of shape (nrow)
    """
    nrow = u.size
    if fourier:
        if delta_x is None:
            raise ValueError("You have to pass in a value for delta_x in Fourier mode")
        out = np.zeros(nrow, dtype=np.complex128)
    else:
        out = np.zeros(nrow, dtype=np.float64)
    for row, ui in enumerate(u):
        for n, c in enumerate(coeffs):
            out[row] += c * basis_function(n, ui, beta, fourier=fourier, delta_x=delta_x)
    return out 

#@numba.jit(nogil=True, nopython=True, cache=True)
def shapelet_2d(u, v, coeffs_l, coeffs_m, fourier, delta_x=None, delta_y=None, beta=1.0):
    nrow_u = u.size
    nrow_v = v.size
    if fourier:
        if delta_x is None or delta_y is None:
            raise ValueError("You have to pass in a value for delta_x and delta_y in Fourier mode")
        out = np.zeros((nrow_u, nrow_v), dtype=np.complex128)
    else:
        out = np.zeros((nrow_u, nrow_v), dtype=np.float64)
    for i, ui in enumerate(u):
        for j, vj in enumerate(v):
            for n1, c1 in enumerate(coeffs_l):
                for n2, c2 in enumerate(coeffs_m):
                    out[i, j] += c1 * basis_function(n1, ui, beta, fourier=fourier, delta_x=delta_x) \
                        * c2 * basis_function(n2, vj, beta, fourier=fourier, delta_x=delta_y)
    return out