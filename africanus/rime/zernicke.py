import numba
import numpy as np
from scipy.misc import factorial as fac

@numba.jit
def zernike_rad(self, m, n, rho):
    if (n < 0 or m < 0 or abs(m) > n):
        raise ValueError
    if ((n-m) % 2):
        return rho*0.0
    pre_fac = lambda k: (-1.0)**k * fac(n-k) / ( fac(k) * fac( (n+m)/2.0 - k ) * fac( (n-m)/2.0 - k ) )
    radial_component = 0
    for k in xrange((n-m)/2+1):
        radial_component += pre_fac(k) * rho **(n - 2.0 * k)
    return radial_component

@numba.jit
def zernike(self, m, n, rho, phi):
    if (m > 0): return self.zernike_rad(m, n, rho) * np.cos(m * phi)
    if (m < 0): return self.zernike_rad(-m, n, rho) * np.sin(-m * phi)
    return self.zernike_rad(0, n, rho)

@numba.jit
def _convert_coords(l, m):
    rho, phi = np.sqrt(l * l + m * m), np.tan(m / l)
    return rho, phi

@numba.jit
def zernike_dde(self, coords, m, n):
    """
    Parameters:
    -----------
    coords: source coordinates of shape (3, src, time, ant, chan)
    m and n: the two variables describing the order of the Zernicke function
    Returns:
    --------
    Zernicke polynomial with shape (src, time, ant, chan).
    """
    if len(coords) != 3:
        raise ValueError("coords must be of shape (3, src, time, ant, chan).")
    _, nsrc, ntime, na, nchan = coords.shape
    zernike_coeffs = np.zeros((nsrc, ntime, na, nchan))
    for h in range(nsrc):
        for i in range(ntime):
            for j in range(na):
                for k in range(nchan):
                    l_coord, m_coord = coords[0, h, i, j, k], coords[1, h, i, j, k]
                    rho, phi = _convert_coords(l_coord, m_coord)
                    zernike_value = self.zernike(self, m, n, rho, phi)
                    zernike_coeffs[h,i,j,k] = zernike_value
    return zernike_coeffs