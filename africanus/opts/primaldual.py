from .sub_opts import *


def primal_dual_solver(x_0, v_0, L, LT, solver='rspd', dask=True, uncert=1.0, maxiter=200, tolerance=1e-6, tau=None,
                       sigma=None, llambda=None):

    M = v_0.shape[0]  # dimension of data
    eps = np.sqrt(2 * M + 2 * np.sqrt(4 * M)) * uncert  # the width of the epsilon ball for the data

    if dask:
        L_norm = power_dask(L, LT, x_0.shape)
        l2ball = lambda v_i: da_proj_l2ball(v_i, eps, v_0)
        l1 = lambda x, t: da_proj_l1_plus_pos(x, t)
        differ = lambda x_new, x, n: da_get_diff(x_new, x, n)
    else:
        L_norm = pow_method(L, LT, x_0.shape)
        l2ball = lambda v_i: proj_l2ball(v_i, eps, v_0)
        l1 = lambda x, t: proj_l1_plus_pos(x, t)
        differ = lambda x_new, x, n: get_diff(x_new, x, n)

    if tau is None:
        tau = 0.95/(np.sqrt(L_norm))

    if sigma is None:
        sigma = 0.95/(2*np.sqrt(L_norm))

    if llambda is None:
        llambda = 1

    print("Tau: ", tau, ", Sigma: ", sigma, ", Lambda: ", llambda)

    def fbpd():
        print("Using Forward-Back Primal-Dual")
        x = x_0.copy()
        v = v_0.copy()

        for n in range(maxiter):
            # Calculate x update step
            x_i = x - tau*LT(v)
            p_n = l1(x_i, tau)

            # Calculate v update step
            v_i = v + sigma * L(2*p_n - x)
            q_n = v_i - l2ball(v_i/sigma)

            # Update x and v
            x_new = x + llambda * (p_n - x)
            v_new = v + llambda * (q_n - v)

            diff = differ(x_new, x, n)

            # Set new values to current values
            x = x_new
            v = v_new

            if diff < tolerance:
                break

        return x

    def rescaled_pd():
        print("Using Rescaled Primal-Dual")
        x = x_0.copy()
        v = v_0.copy()/sigma

        for n in range(maxiter):
            # Calculate x update step
            x_i = x - tau*sigma*LT(v)
            p_n = l1(x_i, tau)

            # Calculate v update step
            v_i = v + L(2*p_n - x)
            q_n = v_i - l2ball(v_i)

            # Update x and v
            x_new = x + llambda * (p_n - x)
            v_new = v + llambda * (q_n - v)

            diff = differ(x_new, x, n)

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
            v_i = v + sigma*L(x)
            q_n = v_i - l2ball(v_i)

            # Calculate v update step
            x_i = x - tau * sigma * LT(2 * q_n - v)
            p_n = l1(x_i, tau)

            # Update x and v
            x_new = x + llambda * (p_n - x)
            v_new = v + llambda * (q_n - v)

            diff = differ(x_new, x, n)

            # Set new values to current values
            x = x_new.compute()
            v = v_new.compute()

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
            q_n = v_i - l2ball(v_i / sigma)

            # Calculate v update step
            x_i = x - tau * LT(2 * q_n - v)
            p_n = l1(x_i, tau)

            # Update x and v
            x_new = x + llambda * (p_n - x)
            v_new = v + llambda * (q_n - v)

            diff = differ(x_new, x, n)

            # Set new values to current values
            x = x_new
            v = v_new

            if diff < tolerance:
                break

        return x

    return {
        'fbpd': lambda: fbpd(),
        'rpd': lambda: rescaled_pd(),
        'spd': lambda: symmetric_pd(),
        'rspd': lambda: rescaled_sym_pd(),
    }[solver]()

