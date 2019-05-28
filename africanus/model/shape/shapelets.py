
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
        coeffs: shapelet coefficients with shape (nsrc, nmax1, nmax2)
        beta: characteristic shapelet size with shape (nsrc, 2)
    Returns:
        out_shapelets: Shapelet with shape (nsrc, nrow)
    """
    fwhmint = 1.0 / np.sqrt(np.log(256))
    gauss_scale = fwhmint * np.sqrt(2.) * (np.pi / lightspeed)
    nrow, _ = coords.shape
    nsrc, nmax1, nmax2 = coeffs.shape
    nchan = coeffs.shape[0]
    #delta_x = np.min(1/(2 * np.max(coords[:, 0])), 1/(2 * np.max(coords[:, 1])))
    max_u = np.max(coords[:, 0])
    max_v = np.max(coords[:, 1])
    print("ABOUT TO CALCULATE DELTA_X")
    delta_x = 1/(2 * max_u) if max_u > max_v else 1/(2 * max_v)
    x_range = [-3 * np.sqrt(2) * (beta[0,0] ** (-1)), 3 * np.sqrt(2) * (beta[0,0] ** (-1))]
    y_range = [-3 * np.sqrt(2) * (beta[0,0] ** (-1)), 3 * np.sqrt(2) * (beta[0,1] ** (-1))]
    npix_x = int((x_range[1] - x_range[0]) / delta_x)
    npix_y = int((y_range[1] - y_range[0]) / delta_x)
    l_vals = np.linspace(x_range[0], x_range[1], npix_x)
    m_vals = np.linspace(y_range[0], y_range[1], npix_y)
    print("CALCULATED DELTA X")
    print(l_vals.shape)
    ll, mm = np.meshgrid(l_vals, m_vals)
    print("CREATED MESHGRID")
    lm = np.vstack((ll.flatten(), mm.flatten()))
    #print(lm.shape)

    
    out_shapelets = np.empty((nsrc, lm.shape[1], nchan), dtype=np.complex128)
    print("LOOPING NOW")
    for i in range(lm.shape[1]):
        for chan in range(nchan):
            #fu = u * frequency[chan] * gauss_scale
            #fv = v * frequency[chan] * gauss_scale 
            #l = convert_to_lm(fu, nyquist_frequency, u_min, u_max)
            #m = convert_to_lm(fv, nyquist_frequency, v_min, v_max)
            #print((u, fu, gauss_scale, frequency[chan]))
            #print((v, fv, gauss_scale, frequency[chan]))
            l = lm[0, i]
            m = lm[1, i]
            for src in range(nsrc):
                beta_u, beta_v = beta[src, :]
                tmp_shapelet = 0 #np.complex128(0. + 0.j)
                for n1 in range(nmax1):
                    for n2 in range(nmax2):
                        sl_val = coeffs[src, n1, n2]
                        #print("sl_val_1 is %f" %sl_val)
                        #sl_val *= (1j**(n1)) * (1j ** (n2))
                        #print("sl_val_2 is %f" %sl_val)
                        sl_val *= basis_function(n1, l, beta_u)
                        #print("*************************************************")
                        #print("sl_val_3 is %f, with u value %f" %(sl_val, u))
                        sl_val *= basis_function(n2, m, beta_v)
                        #print("sl_val_4 is %f, with v value %f" %(sl_val, v))
                        tmp_shapelet += sl_val
                        #print((i, chan, src, n1, n2))
                #print("tmp_shapelet is %f" %tmp_shapelet)
                out_shapelets[src, i, chan] = tmp_shapelet
    print("Finished shapelets")
    return out_shapelets


"""
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
    
    shapelet: computes the shapelet model image in Fourier space
    Inputs:
        coords: coordinates in (u,v) space with shape (nrow, 3)
        coeffs: shapelet coefficients with shape (nsrc, nmax1, nmax2)
        beta: characteristic shapelet size with shape (nsrc, 2)
    Returns:
        out_shapelets: Shapelet with shape (nsrc, nrow)
    
    fwhmint = 1.0 / np.sqrt(np.log(256))
    gauss_scale = fwhmint * np.sqrt(2.) * (np.pi / lightspeed)
    nrow, _ = coords.shape
    nsrc, nmax1, nmax2 = coeffs.shape
    nchan = coeffs.shape[0]
    out_shapelets = np.empty((nsrc, nrow, nchan), dtype=np.complex128)
    #delta_x = 1 / (2 * np.max(np.max(coords[:, 0]), np.max(coords[:, 1])))
    print("*******************")
    for row in range(nrow):
        u, v, w = coords[row, :]
        for chan in range(nchan):
            fu = u * frequency[chan] * gauss_scale
            fv = v * frequency[chan] * gauss_scale 
            #l = convert_to_lm(fu, nyquist_frequency, u_min, u_max)
            #m = convert_to_lm(fv, nyquist_frequency, v_min, v_max)
            #print((u, fu, gauss_scale, frequency[chan]))
            #print((v, fv, gauss_scale, frequency[chan]))
            for src in range(nsrc):
                beta_u, beta_v = beta[src, :]
                tmp_shapelet = np.complex128(0. + 0.j)
                for n1 in range(nmax1):
                    for n2 in range(nmax2):
                        sl_val = coeffs[src, n1, n2] + 0j
                        #print("sl_val_1 is %f" %sl_val)
                        sl_val *= (1j**(n1)) * (1j ** (n2))
                        #print("sl_val_2 is %f" %sl_val)
                        sl_val *= basis_function(n1, fu, beta_u ** (-1))
                        #print("sl_val_3 is %f, with u value %f" %(sl_val, u))
                        sl_val *= basis_function(n2, fv, beta_v ** (-1))
                        #print("sl_val_4 is %f, with v value %f" %(sl_val, v))
                        tmp_shapelet += sl_val
                #print("tmp_shapelet is %f" %tmp_shapelet)
                out_shapelets[src, row, chan] = tmp_shapelet
    return out_shapelets
"""