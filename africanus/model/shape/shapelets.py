import numba
import numpy as np
import numpy.polynomial.hermite as hermite
from numpy import sqrt, exp
from scipy.special import hermite

e = 2.7182818284590452353602874713527
square_root_of_pi = 1.77245385091

#@numba.jit(nogil=True, nopython=True, cache=True)
def factorial(n):
    if n <= 1:
        return 1
    ans = 1
    for i in range(1, n):
        ans *= i
    return ans * n

#@numba.jit(nogil=True, nopython=True, cache=True)
def basis_function(n, x, beta):
    basis_component = ((2**n) * ((np.pi)**(0.5)) * factorial(n) * beta)**(-0.5)
    exponential_component = hermite(n)(x / beta) * np.exp((-0.5) * (x**2) * (beta **(-2)))
    return basis_component * exponential_component

#@numba.jit(nogil=True, nopython=True, cache=True)
def shapelet(coords, coeffs, beta):
    """
    shapelet: computes the shapelet model image in Fourier space
    Inputs:
        coords: coordinates in (u,v) space with shape (nrow, 3)
        coeffs: shapelet coefficients with shape (nsrc, nmax1, nmax2)
        beta: characteristic shapelet size with shape (nsrc, 2)
    Returns:
        out_shapelets: Shapelet with shape (nsrc, nrow)
    """
    nrow, _ = coords.shape
    nsrc, nmax1, nmax2 = coeffs.shape
    out_shapelets = np.empty((nsrc, nrow), dtype=np.complex128)
    for row in range(nrow):
        u, v, w = coords[row, :]
        for src in range(nsrc):
            beta_u, beta_v = beta[src, :]
            tmp_shapelet = 0 + 0j
            for n1 in range(nmax1):
                for n2 in range(nmax2):
                    sl_val = coeffs[src, n1, n2] + 0j
                    sl_val *= (1j**(n1)) * (1j ** (n2))
                    sl_val *= basis_function(n1, u, beta_u ** (-1))
                    sl_val *= basis_function(n2, v, beta_v ** (-1))
                    tmp_shapelet += sl_val
            out_shapelets[src, row] = tmp_shapelet
    return out_shapelets
