import xarrayms
from africanus.dft.kernels import im_to_vis, vis_to_im
from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
from casacore.tables import table as tbl

import numba
import numpy as np
from .sub_opts import *


def primal_dual_solver(x_0, v_0, L, LT, wsum, solver='fbpd', uncert=1.0, maxiter=2000, tolerance=1e-6, tau=None, sigma=None,
                       llambda=None):

    L_norm = pow_method(L, LT, [x_0.shape[0], 1])
    print('The norm of the response: ', L_norm)

    if tau is None:
        tau = 1.0/(2*np.sqrt(L_norm))

    if sigma is None:
        sigma = 1.0/np.sqrt(L_norm)

    if llambda is None:
        llambda = 1

    print("Tau: ", tau, ", Sigma: ", sigma, ", Lambda: ", llambda)

    M = v_0.size  # dimension of data
    eps = np.sqrt(2 * M + 2 * np.sqrt(4 * M)) * uncert  # the width of the epsilon ball for the data

    # print({
    #     'fbpd': "Using Forward-Back Primal-Dual",
    #     'rpd': "Using Rescaled Primal-Dual",
    #     'spd': "Using Symmetric Primal-Dual",
    # }[solver])

    def fbpd(x, v):
        # Calculate x update step
        x_i = x - tau*LT(v)
        p_n = proj_l1_plus_pos(x_i, tau)

        # Calculate v update step
        v_i = v + sigma * L(2 * p_n - x)
        q_n = proj_l2ball(v_i, eps, v)

        return x + llambda * (p_n - x), v + llambda * (q_n - v)

    def rescaled_pd(x, v):
        # Calculate x update step
        x_i = x - tau*sigma*LT(v)
        p_n = proj_l1_plus_pos(x_i, tau)

        # Calculate v update step
        v_i = v + L(2*p_n - x)
        q_n = proj_l2ball(v_i, eps, v)

        # Update x and v
        return x + llambda * (p_n - x), v + llambda * (q_n - v)

    def rescaled_sym_pd():
        print("Using Rescaled Symmetric Primal Dual")
        x = x_0.copy()
        v = v_0.copy()

        for n in range(maxiter):
            # Calculate x update step
            v_i = v + sigma * L(x)
            q_n = v_i - proj_l2ball(v_i, eps, v_0)

            # Calculate v update step
            x_i = x - tau * sigma * LT(2 * q_n - v) / wsum
            p_n = proj_l1_plus_pos(x_i, tau)

            # Update x and v
            x_new = x + llambda * (p_n - x)
            v_new = v + llambda * (q_n - v)


            diff = get_diff(x_new, x, n)

            # Set new values to current values
            x = x_new
            v = v_new

            if diff < tolerance:
                break

        return x

    def symmetric_pd(x, v):
        # Calculate v update step
        v_i = v + sigma * L(x)
        q_n = proj_l2ball(v_i / sigma, eps, v_0)

        # Calculate v update step
        x_i = x - tau * LT(2 * q_n - v) / wsum
        p_n = proj_l1_plus_pos(x_i, tau)

        #return x + llambda*(p_n - x), v + llambda*(q_n - v)
        return p_n, q_n

    return rescaled_sym_pd()
    #     {
    #     'fbpd': lambda: fbpd(x, v),
    #     'rpd': lambda: rescaled_pd(x, v),
    #     'spd': lambda: symmetric_pd(x, v),
    # }[solver]()

def get_diff(x_new, x, n):

    # Get new norms
    norm2 = np.linalg.norm(x_new)
    norm1 = np.linalg.norm(x_new, 1)

    # get diff i.t.o. 1-norm
    diff1 = np.linalg.norm(x_new - x, 1) / norm1
    # get diff i.t.o. 2-norm
    diff2 = np.linalg.norm(x_new - x) / norm2

    print('L1 norm=', norm1, ' L2 norm=', norm2)
    print('Iter = %i, diff1 = %f, diff2 = %f' % (n, diff1, diff2))

    return np.maximum(diff1, diff2)
