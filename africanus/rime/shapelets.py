import numba
import numpy as np
import math
import numpy.polynomial.hermite as hermite

def factorial(n):
    if n == 0 or n == 1:
        return 1
    ret_val = 1
    for i in range(1, n):
        ret_val *= i
    return ret_val


@numba.jit(nogil=True, nopython=True, cache=True)
def basis_function(n, x, beta):
    return (1 / sqrt(pow(2, n) * sqrt(np.pi) * factorial(n) * beta)) * hermite.hermval(x / beta, n) * exp(-(x ** 2) / (2 * beta * beta))

@numba.jit(nogil=True, nopython=True, cache=True)
def shapelet_dde(coords, coeff_beta, shapelet_coeffs):
    _, nsrc, ntime, na, nchan = coords.shape
    _, _, npoly = coeffs.shape
    out_shapelets = np.empty((nsrc, ntime, na, nchan), dtype=np.complex128)
    for src in range(nsrc):
        for time in range(ntime):
            for ant in range(na):
                for chan in range(nchan):
                    l, m = coords[:, src, time, ant, chan]
                    coeff_subset = coeffs[ant, chan]
                    shapelet_sum = 0
                    beta = coeff_beta[ant, chan]
                    for poly in range(npoly):
                        beta, f = coeff_subset[poly]
                        shapelet_sum += basis_function(poly, l, beta) * basis_function(poly, m, beta)
                    out_shapelets[src, time, ant, chan] = shapelet_sum
    return out_shapelets