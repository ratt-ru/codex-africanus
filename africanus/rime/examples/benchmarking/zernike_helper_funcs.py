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
    # if numerator/denominator > 10:
    #     print(numerator/denominator, n, k, m)
    # quit()
    return numerator / denominator



def zernike_rad(m_indices, n_indices, rho):
    radial_component = np.zeros(rho.shape)
    # print(m_indices.shape, rho.shape)
    # quit()
    for c in range(m_indices.shape[0]):
        n, m = n_indices[c], m_indices[c]
        # print(m.shape)
        # quit()
        for k in range(int((n-m)/2 + 1)):
            radial_component += pre_fac(k, n, m) * rho ** (n - 2.0 * k)
    # print(np.where(radial_component == 32))
    # quit()
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

def _fac(x):
    if x < 0:
        raise ValueError("Factorial input is negative.")
    if x == 0:
        return 1
    factorial = 1
    for i in range(1, int(x + 1)):
        factorial *= i
    return factorial

def _pre_fac(k, n, m):
    numerator = (-1.0)**k * _fac(n-k)
    denominator = (_fac(k) * _fac((n+m)/2.0 - k) * _fac((n-m)/2.0 - k))
    return numerator / denominator


def _zernike_rad(m, n, rho):
    if (n < 0 or m < 0 or abs(m) > n):
        raise ValueError("m and n values are incorrect. m = ", m , " and n = ", n)
    radial_component = 0
    # print(n,m,rho)
    # quit()
    # print("Starting loop now")
    for k in range(int((n-m)/2+1)):
        radial_component += _pre_fac(k, n, m) * rho ** (n - 2.0 * k)
        # print(radial_component)
    # print("###################")
    # if radial_component > 10: print(radial_component)
    return radial_component

def zernike_func(n, m, rho, phi):
    #Per basis function
    # output_arr = np.empty(rho.shape)
    # print(n.shape)
    # print(rho.shape)
    # quit()
    # for a in range(output_arr.shape[0]):
    #     for b in range(output_arr.shape[1]):
    #         for c in range(output_arr.shape[2]):
    #             for d in range(output_arr.shape[3]):
    #                 if rho[a,b,c,d] > 1: output_arr[a,b,c,d] = 0
    #                 elif m[0,d] > 0: output_arr[a,b,c,d] = _zernike_rad(m[0,d],n[0,d],rho[a,b,c,d])* np.cos(m[0,d] * phi[a,b,c,d])
    #                 elif m[0,d] < 0: output_arr[a,b,c,d] = _zernike_rad(-m[0,d],n[0,d],rho[a,b,c,d])* np.sin(-m[0,d] * phi[a,b,c,d])
    #                 else: output_arr[a,b,c,d] = _zernike_rad(0,n[0,d],rho[a,b,c,d])
                    # if rho 
                    # print(rho[a,b,c,d], m[0,d], output_arr[a,b,c,d])
                    # quit()
    def _vectorized_zernike_func(n,m, rho, phi):
        output_arr = np.empty(rho.shape)
        output_arr[:,0,0,(m<0)[0,:]] = zernike_rad(-1 * m[m<0], n[m<0], rho[:,0,0,(m<0)[0,:]]) * np.sin(-1 * m[m<0] * phi[:,0,0,(m<0)[0,:]])
    
        output_arr[:,0,0,(m>0)[0,:]] = zernike_rad(m[m>0], n[m>0], rho[:,0,0,(m>0)[0,:]]) * np.cos(m[m>0] * phi[:,0,0,(m>0)[0,:]])
        # output_arr[:,0,0,(m==0)[0,:]] = zernike_rad(np.zeros(n[m==0].shape), n[m==0], rho[:,0,0,(m==0)[0,:]]) 
        output_arr[:,0,0,(m==0)[0,:]] = zernike_rad(np.zeros(m[m==0].shape), n[m==0], rho[:,0,0,(m==0)[0,:]])

        output_arr[rho > 1] = 0

        return output_arr  
    v_zernike = _vectorized_zernike_func(n,m,rho,phi)
    v_zernike = v_zernike / np.max(np.abs(v_zernike))
    # print(output_arr[:,:,:,m[0,:] == 0].shape)
    # quit()
    # print(output_arr[:,0,0,m[0,:] == 0].flatten()[140], v_zernike[:,0,0,m[0,:] == 0].flatten()[140])
    # plt.figure()
    # plt.plot(output_arr.flatten(), label="Sequential")
    # plt.plot(v_zernike.flatten(), label="Vectorized")
    # # plt.plot(v_zernike[:,0,0,m[0,:] == 0].flatten()/output_arr[:,0,0,m[0,:] == 0].flatten(), label="Quotient")
    # plt.legend(loc='upper left')
    # plt.show()
    # plt.close()
    # # quit()
    # print(np.allclose(output_arr[:,:,:,m[0,:] == 0] , v_zernike[:,:,:,m[0,:] == 0]))
    # # print(v_zernike[rho > 1])
    # quit() 
    return v_zernike



def _convert_coords(l, m):
    rho, phi = (np.sqrt(l*l + m * m)), np.arctan2(l, m)
    return rho, phi
