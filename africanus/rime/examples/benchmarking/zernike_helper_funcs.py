import numpy as np
import matplotlib.pyplot as plt

def fac(x):
    x = int(x)
    if x < 0:
        raise ValueError("Factorial input is negative.")
    if x == 0:
        return 1
    factorial = 1
    for i in range(1, x + 1):
        factorial *= i
    return factorial


def pre_fac(k, n, m):
    # Per basis function, frequency. Returns a scalar to multiply with all coords
    numerator = (-1.0)**k * fac(n-k)
    denominator = fac(k) * fac((n+m)/2.0 - k) * fac((n-m)/2.0 - k)
    return numerator / denominator



def zernike_rad(m_indices, n_indices, rho):
    radial_component = np.zeros(rho.shape)
    for c in range(m_indices.shape[0]):
        n, m = n_indices[c], m_indices[c]
        for k in range(int((n-m)/2 + 1)):
            radial_component[:,c] += pre_fac(k, n, m) * rho[:,c] ** (n - 2.0 * k)
    return radial_component 



def noll_to_zern(j):
    j = j + 1
    n = np.zeros(j.shape)
    j1 = j - 1
    while ((j1 > n).any()):
        n[j1 > n] = n[j1 > n] + 1
        j1[j1 >= n] = j1[j1 >= n] - n[j1 >= n]
    tmp = np.vectorize(int)((j1+((n+1) % 2)) / 2.0)
    m = (-1)**j * ((n % 2) + 2 * tmp)
    return n, m


def zernike_func(n, m, rho, phi):
    output_arr = np.empty(rho.shape)

    output_arr[:,0,0,(m==0)[0,:]] = zernike_rad(np.zeros(m[m==0].shape), n[m==0], rho[:,0,0,(m==0)[0,:]])
    output_arr[:,0,0,(m<0)[0,:]] = zernike_rad(-1 * m[m<0], n[m<0], rho[:,0,0,(m<0)[0,:]]) * np.sin(-1 * m[m<0] * phi[:,0,0,(m<0)[0,:]])
    output_arr[:,0,0,(m>0)[0,:]] = zernike_rad(m[m>0], n[m>0], rho[:,0,0,(m>0)[0,:]]) * np.cos(m[m>0] * phi[:,0,0,(m>0)[0,:]])

    output_arr[rho > 1] = 0

    return output_arr  


def _convert_coords(l, m):
    rho, phi = (np.sqrt(l*l + m * m)), np.arctan2(l, m)
    return rho, phi
