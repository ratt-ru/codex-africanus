import numba
import numpy as np
import numpy.polynomial.hermite as hermite
from numpy import sqrt, exp
from scipy.special import hermite

e = 2.7182818284590452353602874713527
square_root_of_pi = 1.77245385091

# @numba.jit(nogil=True, nopython=True, cache=True)
def factorial(n):
    if n <= 1:
        return 1
    ans = 1
    for i in range(1, n + 1):
        ans *= i
    return ans


# @numba.jit(nogil=True, nopython=True, cache=True)
def basis_function(n, x, beta):
    basis_component = (
        (2 ** n) * ((np.pi) ** (0.5)) * factorial(n) * beta
    ) ** (-0.5)
    exponential_component = hermite(n)(x / beta) * np.exp(
        (-0.5) * (x ** 2) * (beta ** (-2))
    )
    return basis_component * exponential_component


# @numba.jit(nogil=True, nopython=True, cache=True)
def shapelet_dde(coords, coeffs, beta):
    """
    shapelet_dde: computes the shapelet in Fourier space
    Inputs:
        coords: coordinates in (u,v) space with shape (2, nsrc, ntime, na, nchan)
        coeffs: shapelet coefficients with shape (ant, chan, nmax1, nmax2)
        beta: characteristic shapelet size with shape (ant, chan, 2)
    Returns:
        out_shapelets: Shapelet with shape (nsrc, ntime, na, nchan)
    """
    _, nsrc, ntime, na, nchan = coords.shape
    _, _, nmax1, nmax2 = coeffs.shape
    out_shapelets = np.empty((nsrc, ntime, na, nchan), dtype=np.complex128)
    for ant in range(na):
        for chan in range(nchan):
            betax = beta[ant, chan, 0]
            betay = beta[ant, chan, 1]
            for src in range(nsrc):
                for time in range(ntime):
                    u = coords[0, src, time, ant, chan]
                    v = coords[1, src, time, ant, chan]
                    shapelet_sum = 0
                    for n1 in range(nmax1):
                        for n2 in range(nmax2):
                            # print("n1 is %d and n2 is %d" %(n1, n2))
                            tmp_ans = coeffs[ant, chan, n1, n2] + 0j
                            tmp_ans *= (1j ** (n1)) * (1j ** (n2))
                            tmp_ans *= basis_function(n1, u, betax ** (-1))
                            tmp_ans *= basis_function(n2, v, betay ** (-1))
                            shapelet_sum += tmp_ans
                        # shapelet_sum += coeffs[ant, chan, n1, n2] * 1j**(n1 + n2) * basis_function(n1, u, 1/betax) * basis_function(n2, v, 1/betay)
                    out_shapelets[src, time, ant, chan] = shapelet_sum
    return out_shapelets
