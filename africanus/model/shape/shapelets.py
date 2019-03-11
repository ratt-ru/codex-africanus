import numba
import numpy as np
import numpy.polynomial.hermite as hermite
from numpy import sqrt, exp
from scipy.special import hermite 

e = 2.7182818284590452353602874713527
square_root_of_pi = 1.77245385091

"""
@numba.jit(nogil=True, nopython=True, cache=True)
def hermite(n, x):
    if n==0:
        return 1
    elif n==1:
        return 2*x
 #   else:
 #       return 2*x*hermite(x,n-1)-2*(n-1)*hermite(x,n-2)
    Hn1 = 2 * x
    Hn2 = 1
    Hn = 1
    for i in range(2, n):
        Hn *= 2 * x * Hn1 - 2 * (n-1) * Hn2
        Hn2 = Hn1
        Hn1 = Hn
    return Hn
   """     

@numba.jit(nogil=True, nopython=True, cache=True)
def factorial(n):
    if n <= 1:
        return 1
    ans = 1
    for i in range(1, n + 1):
        ans *= i
    return ans

#@numba.jit(nogil=True, nopython=True, cache=True)
def basis_function(n, x, beta):
    basis_component = ((2**n) * ((np.pi)**(0.5)) * factorial(n) * beta)**(-0.5)
    exponential_component = hermite(n)(x / beta) * np.exp((-0.5) * (x**2) * (beta **(-2)))
    return basis_component * exponential_component

#@numba.jit(nogil=True, nopython=True, cache=True)
def shapelet_dde(coords, coeffs, beta):
    """
    shapelet_dde: computes the shapelet in Fourier space
    Inputs:
        coords: coordinates in (u,v, w) space with shape (nrow, 3)
        coeffs: shapelet coefficients with shape (nsrc, nmax1, nmax2)
        beta: characteristic shapelet size with shape (nsrc, 2), where beta[src, 0] denotes the beta for the u coordinate, and likewise with beta[src, 1]
    Returns:
        out_shapelets: Shapelet with shape (nsrc, nrow)
    """
    nrow, _ = coords.shape
    nsrc, nmax1, nmax2 = coeffs.shape
    out_shapelets = np.zeros((nsrc, nrow), dtype=np.complex128)
    
    for row in range(nrow):
        u, v, _ = coords[row, :]
        for src in range(nsrc):
            beta1, beta2 = beta[src, 0] ** -1, beta[src, 1] ** -1
            for n1 in range(nmax1):
                for n2 in range(nmax2):
                    out_shapelets[src, row] += 1j**(n1 + n2) * coeffs[src, n1, n2] * basis_function(n1, u, beta1) * basis_function(n2, v, beta2)
    return out_shapelets