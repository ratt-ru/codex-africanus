import dask.array as da
from .sub_opts import *


def primal_dual_solver(x_0, v_0, L, LT, solver='fbpd', uncert=1.0, maxiter=2000, tolerance=1e-6, tau=None, sigma=None,
                       llambda=None):

    L_norm = power_dask(L, LT, [x_0.shape[0], 1])
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

    def fbpd():
        print("Using Forward-Back Primal-Dual")
        x = x_0.copy()
        v = v_0.copy()

        for n in range(maxiter):
            # Calculate x update step
            x_i = x - tau*LT(v)
            p_n = proj_l1_plus_pos(x_i, tau)

            # Calculate v update step
            v_i = v + sigma * L(2 * p_n - x)
            q_n = proj_l2ball(v_i, eps, v)

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

    def rescaled_pd():
        print("Using Rescaled Primal-Dual")
        x = x_0.copy()
        v = v_0.copy()

        for n in range(maxiter):
            # Calculate x update step
            x_i = x - tau*sigma*LT(v)
            p_n = proj_l1_plus_pos(x_i, tau)

            # Calculate v update step
            v_i = v + L(2*p_n - x)
            q_n = proj_l2ball(v_i, eps, v)

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

    def rescaled_sym_pd():
        print("Using Rescaled Symmetric Primal Dual")
        x = x_0.copy()
        v = v_0.copy()

        for n in range(maxiter):
            # Calculate x update step
            v_i = v + sigma * L(x)
            q_n = v_i - proj_l2ball(v_i, eps, v_0)

            # Calculate v update step
            x_i = x - tau * sigma * LT(2 * q_n - v)
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

    def symmetric_pd():
        print("Using Symmetric Primal-Dual")
        x = x_0.copy()
        v = v_0.copy()

        for n in range(maxiter):
            # Calculate v update step
            v_i = v + sigma * L(x)
            q_n = proj_l2ball(v_i / sigma, eps, v_0)

            # Calculate v update step
            x_i = x - tau * LT(2 * q_n - v)
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

    return  {
        'fbpd': lambda: fbpd(),
        'rpd': lambda: rescaled_pd(),
        'spd': lambda: symmetric_pd(),
        'rspd': lambda: rescaled_sym_pd(),
    }[solver]()


def get_diff(x_new, x, n):

    # Get new norms
    norm2 = da.linalg.norm(x_new).compute()
    norm1 = da.linalg.norm(x_new, 1).compute()

    # get diff i.t.o. 1-norm
    diff1 = da.linalg.norm(x_new - x, 1).compute() / norm1
    # get diff i.t.o. 2-norm
    diff2 = da.linalg.norm(x_new - x).compute() / norm2

    print('L1 norm=', norm1, ' L2 norm=', norm2)
    print('Iter = %i, diff1 = %f, diff2 = %f' % (n, diff1, diff2))

    return da.maximum(diff1, diff2)
