import numba
import numpy as np

@numba.jit(nogil=True, nopython=True)
def fac(x):
    if x < 0: return -1 * x
    if x == 0: return 1
    factorial = 1
    for i in range(1, x):
        factorial *= i
    return factorial

@numba.jit(nogil=True, nopython=True)
def pre_fac(k, n, m):
    return (-1.0)**k * fac(n-k) / ( fac(k) * fac( (n+m)/2.0 - k ) * fac( (n-m)/2.0 - k ))

@numba.jit(nogil=True, nopython=True)
def zernike_rad( m, n, rho):
    if (n < 0 or m < 0 or abs(m) > n):
        raise ValueError
    if ((n-m) % 2):
        return rho*0.0
        
    radial_component = 0
    for k in range((n-m)/2+1):
        radial_component = radial_component + pre_fac(k, n, m) * rho **(n - 2.0 * k)
    return radial_component

@numba.jit(nogil=True, nopython=True)
def zernike(j, rho, phi):
    j += 1
    n = 0
    j1 = j-1
    while (j1 > n):
        n += 1
        j1 -= n
    m = (-1)**j * ((n % 2) + 2 * int((j1+((n+1)%2)) / 2.0 ))
    if (m > 0): return zernike_rad(m, n, rho) * np.cos(m * phi)
    if (m < 0): return zernike_rad(-m, n, rho) * np.sin(-m * phi)
    return zernike_rad(0, n, rho)

@numba.jit(nogil=True, nopython=True)
def _convert_coords(l, m):
    rho, phi = np.sqrt(l * l + m * m), np.arctan(m / l)
    return rho, phi

@numba.jit(nogil=True, nopython=True)
def zernike_dde(coords, coeffs, noll_index):
    """
    Evaluate Zernicke Polynomials defined by coefficients ``coeffs`
    at the specified coordinates ``coords``.

    Parameters
    ---------------
    coords : :class:`numpy.ndarray`
       Coordinates at which to evaluate the zernicke polynomials.
       Has shape :code:`(3, source, time, ant, chan)`. The three components in
       the first dimension represent l, m and frequency coordinates, respectively.
    coeffs : :class:`numpy.ndarray`
      Zernicke polynomial coefficients.
      Has shape :code:`(ant, chan, poly)` where ``poly`` is the number of
      polynomial coefficients.
    noll_index : :class:`numpy.ndarray`
      Noll index associated with each polynomial coefficient.
      Has shape :code:`(ant, chan, poly)`.

    Returns
    ----------
    :class:`numpy.ndarray`
       complex values with shape :code:`(source, time, ant, chan)`
    """
    _, sources, times, ants, chans = coords.shape
    _,_, npoly = coeffs.shape

    result = np.empty((sources,times,ants,chans), np.complex128)

    for s in range(sources):
        for t in range(times):
            for a in range(ants):
                for c in range(chans):
                    l, m, freq = coords[:,s,t,a,c]
                    rho, phi = _convert_coords(l,m)
                    zcoeff = coeffs[a,c]
                    zernike_sum = 0
                    for i in range(npoly):
                        coeff = zcoeff[i]
                        j = noll_index[a,c,i]
                        zernike_sum += coeff * zernike(j, rho, phi)
                    result[s,t,a,c] = zernike_sum   
    return result