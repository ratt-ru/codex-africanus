import numba
import numpy as np
from numpy import sqrt, exp
from africanus.constants import c as lightspeed

e = 2.7182818284590452353602874713527
square_root_of_pi = 1.77245385091

#@numba.jit(nogil=True, nopython=True, cache=True)
def hermite(n, x):
    if n==0:
        return 1
    elif n==1:
        return 2*x
    else:
        return 2*x*hermite(x,n-1)-2*(n-1)*hermite(x,n-2)


#@numba.jit(nogil=True, nopython=True, cache=True)
def factorial(n):
    if n <= 1:
        return 1
    ans = 1
    for i in range(1, n):
        ans *= i
    return ans * n

#@numba.jit(nogil=True, nopython=True, cache=True)
def basis_function(n, xx, beta):
    basis_component = ((2**n) * ((np.pi)**(0.5)) * factorial(n) * beta)**(-0.5)
    exponential_component = hermite(n, xx / beta) * np.exp((-0.5) * (xx**2) * (beta **(-2)))
    return basis_component * exponential_component


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
            fu = u# * frequency[chan]# * gauss_scale
            fv = v# * frequency[chan]# * gauss_scale
            #print(fu, fv, frequency[chan])
            #print(fu, fv)
            for src in range(nsrc):
                beta_u, beta_v = beta[src, :] ** (-1)
                tmp_shapelet = 0 + 0j
                for n1 in range(nmax1):
                    for n2 in range(nmax2):
                        tmp_shapelet += coeffs[src, n1, n2] * basis_function(n1, fu, beta_u) * basis_function(n2, fv, beta_v) if (n1 + n2 % 4) == 0\
                            else -1 * coeffs[src, n1, n2] * basis_function(n1, fu, beta_u) * basis_function(n2, fv, beta_v) if (n1 + n2 % 4) == 2 \
                            else 1j * coeffs[src, n1, n2] * basis_function(n1, fu, beta_u) * basis_function(n2, fv, beta_v)
                        #print(tmp_shapelet)
                #print("tmp_shapelet is %f" %tmp_shapelet)
                out_shapelets[src, row, chan] = tmp_shapelet
    return out_shapelets