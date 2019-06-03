import numba
import numpy as np
import numpy.polynomial.hermite as hermite
from numpy import sqrt, exp
from scipy.special import hermite
from africanus.constants import c as lightspeed

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
    #print("basis_component is %f" %basis_component)
    exponential_component_1 = hermite(n)(x / beta)
    #print("exponential_component_1 is %f" %exponential_component_1)
    exponential_component_2 = np.exp((-0.5) * (x**2) * (beta **(-2)))
    #print("exponential_component_2 is %f" %exponential_component_2)
    exponential_component = exponential_component_1 * exponential_component_2
    return basis_component * exponential_component

def convert_to_lm(u, nyquist_frequency, u_min, u_max):
    return nyquist_frequency * ((u - u_min) / (u_max - u_min))

#@numba.jit(nogil=True, nopython=True, cache=True)
def shapelet(coords, frequency, coeffs, beta):
    """
    shapelet: computes the shapelet model image in Fourier space
    Inputs:
        coords: coordinates in (u,v) space with shape (nrow, 3)
        frequency: frequency values with shape (nchan,)
        coeffs: shapelet coefficients with shape (nsrc, nmax1, nmax2)
        beta: characteristic shapelet size with shape (nsrc, 2)
    Returns:
        out_shapelets: Shapelet with shape (nsrc, nrow, nchan)
    """
    
    fwhmint = 1.0 / np.sqrt(np.log(256))
    gauss_scale = fwhmint * np.sqrt(2.) * (np.pi / lightspeed)
    nrow, _ = coords.shape
    nsrc, nmax1, nmax2 = coeffs.shape
    nchan = coeffs.shape[0]
    out_shapelets = np.empty((nsrc, nrow, nchan), dtype=np.complex128)
    for row in range(nrow):
        u, v, w = coords[row, :]
        for chan in range(nchan):
            fu = u * frequency[chan] * gauss_scale
            fv = v * frequency[chan] * gauss_scale 
            for src in range(nsrc):
                beta_u, beta_v = beta[src, :]
                tmp_shapelet = np.complex128(0. + 0.j)
                for n1 in range(nmax1):
                    for n2 in range(nmax2):
                        tmp_shapelet += coeffs[src, n1, n2] * (1j**(n1)) * (1j ** (n2)) * basis_function(n1, fu, beta_u ** (-1)) * basis_function(n2, fv, beta_v ** (-1))
                #print("tmp_shapelet is %f" %tmp_shapelet)
                out_shapelets[src, row, chan] = tmp_shapelet
    return out_shapelets