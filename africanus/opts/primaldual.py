import numba
import numpy as np
import scipy.sparse.linalg as sla
from .sub_opts import proj_l1_plus_pos, proj_l2ball, pow_method


def primal_dual_solver(x_0, v_0, L, LT, solver='fbpd', uncert=1.0, maxiter=1000, tolerance=1e-3, tau=None, sigma=None,
                       llambda=None):

    npix = int(np.sqrt(x_0.shape[0]))

    L_norm = pow_method(L, LT, [npix, npix])

    if tau is None:
        tau = 1.0/L_norm

    if sigma is None:
        sigma = 1.0/L_norm

    if llambda is None:
        llambda = 1.0

    M = v_0.size  # dimension of data
    eps = np.sqrt(2 * M + 2 * np.sqrt(4 * M)) * uncert  # the width of the epsilon ball for the data

    x = x_0.flatten()
    v = np.zeros(M, dtype=np.complex128)

    def fbpd(x):
        for n in range(maxiter):
            # Calculate x update step
            x_i = x - tau*LT(v)
            p_n = proj_l1_plus_pos(x_i, tau)

            # Calculate v update step
            v_i = v + sigma*L(2*p_n - x)
            q_n = proj_l2ball(sigma*v_i, eps, v)

            # Update x and v
            x_new = x + llambda(p_n - x)
            v_new = v + llambda(q_n - v)

            # get diff i.t.o. 2-norm
            diff1 = np.linalg.norm(x_new - x) / np.linalg.norm(x_new)
            # get diff i.t.o. 1-norm
            diff2 = np.linalg.norm(x_new - x, 1) / np.linalg.norm(x_new, 1)

            diff = np.maximum(diff1, diff2)

            # Set new values to current values
            x = x_new
            v = v_new

            if diff < tolerance:
                break

            if n%10 == 0:
                print('Iter = %i, diff1 = %f, diff2 = %f' % (n, diff1, diff2))

        print('Final iter = %i, diff1 = %f, diff2 = %f' % (n, diff1, diff2))
        return x

    def rescaled_pd():
        v_p = v/sigma
        for n in range(maxiter):
            # Calculate x update step
            x_i = x - tau*sigma*LT(v_p)
            p_n = proj_l1_plus_pos(x_i, tau)

            # Calculate v update step
            v_i = v_p + L(2*p_n - x)
            q_n = v_p - proj_l2ball(v_i/sigma, eps, v_p)

            # Update x and v
            x_new = x + llambda(p_n - x)
            v_new = v_p + llambda(q_n - v_p)

            # get diff i.t.o. 2-norm
            diff1 = np.linalg.norm(x_new - x) / np.linalg.norm(x_new)
            # get diff i.t.o. 1-norm
            diff2 = np.linalg.norm(x_new - x, 1) / np.linalg.norm(x_new, 1)

            diff = np.maximum(diff1, diff2)

            # Set new values to current values
            x = x_new
            v_p = v_new

            if diff < tolerance:
                break

            if n % 10 == 0:
                print('Iter = %i, diff1 = %f, diff2 = %f' % (n, diff1, diff2))

        print('Final iter = %i, diff1 = %f, diff2 = %f' % (n, diff1, diff2))
        return x

    def symmetric_pd():
        for n in range(maxiter):
            # Calculate v update step
            v_i = v - sigma*LT(x)
            q_n = proj_l2ball(sigma*v_i, eps, v)

            # Calculate v update step
            x_i = x - tau*LT(2*q_n - v)
            p_n = proj_l1_plus_pos(x_i,tau)

            # Update x and v
            x_new = x + llambda(p_n - x)
            v_new = v + llambda(q_n - v)

            # get diff i.t.o. 2-norm
            diff1 = np.linalg.norm(x_new - x) / np.linalg.norm(x_new)
            # get diff i.t.o. 1-norm
            diff2 = np.linalg.norm(x_new - x, 1) / np.linalg.norm(x_new, 1)

            diff = np.maximum(diff1, diff2)

            # Set new values to current values
            x = x_new
            v = v_new

            if diff < tolerance:
                break

            if n%10 == 0:
                print('Iter = %i, diff1 = %f, diff2 = %f' % (n, diff1, diff2))

        print('Final iter = %i, diff1 = %f, diff2 = %f' % (n, diff1, diff2))
        return x

    return {
        'fbpd': fbpd(x),
        'rpd': rescaled_pd(x),
        'spd': symmetric_pd(x),
    }[solver]
