from .sub_opts import *


def primal_dual_solver(x_0, v_0, L, LT, solver='rspd', dask=True, uncert=1.0, maxiter=500, tol=1e-6, tau=None,
                       sigma=None, llambda=None):

    M = v_0.shape[0]  # dimension of data
    eps = np.sqrt(2 * M + 2 * np.sqrt(4 * M)) * uncert  # the width of the epsilon ball for the data

    if dask:
        L_norm = power_dask(L, LT, x_0.shape)
        l2ball = lambda v_i: da_proj_l2ball(v_i, eps, v_0)
        l1 = lambda x, t: da_proj_l1_plus_pos(x, t)
    else:
        L_norm = pow_method(L, LT, x_0.shape)
        l2ball = lambda v_i: proj_l2ball(v_i, eps, v_0)
        l1 = lambda x, t: proj_l1_plus_pos(x, t)

    if tau is None:
        tau = 1.0/(2*(L_norm))

    if sigma is None:
        sigma = .95/((L_norm))

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

            # Get new norms
            norm2 = da.linalg.norm(x_new)
            norm1 = da.linalg.norm(x_new, 1)

            if norm1 == 0:
                norm1 = 1
            if norm2 == 0:
                norm2 = 1

            # get diff i.t.o. 1-norm
            diff1 = da.linalg.norm(x_new - x, 1) / norm1
            # get diff i.t.o. 2-norm
            diff2 = da.linalg.norm(x_new - x) / norm2

            if dask:
                x_new, v_new, norm1, norm2, diff1, diff2 = da.compute(x_new, v_new, norm1, norm2, diff1, diff2)

            print('L1 norm=', norm1, ' L2 norm=', norm2)
            print('Iter = %i, diff1 = %f, diff2 = %f' % (n, diff1, diff2))

            # Set new values to current values
            x = x_new
            v = v_new

            diff = max(diff1, diff2)
            if diff < tol:
                break

        return x_new

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

            # Get new norms
            norm2 = da.linalg.norm(x_new)
            norm1 = da.linalg.norm(x_new, 1)

            if norm1 == 0:
                norm1 = 1
            if norm2 == 0:
                norm2 = 1

            # get diff i.t.o. 1-norm
            diff1 = da.linalg.norm(x_new - x, 1) / norm1
            # get diff i.t.o. 2-norm
            diff2 = da.linalg.norm(x_new - x) / norm2

            if dask:
                x_new, v_new, norm1, norm2, diff1, diff2 = da.compute(x_new, v_new, norm1, norm2, diff1, diff2)

            print('L1 norm=', norm1, ' L2 norm=', norm2)
            print('Iter = %i, diff1 = %f, diff2 = %f' % (n, diff1, diff2))

            # Set new values to current values
            x = x_new
            v = v_new

            diff = max(diff1, diff2)
            if diff < tol:
                break

        return x_new

    def rescaled_sym_pd():
        print("Using Rescaled Symmetric Primal Dual")
        x = x_0.copy()
        v = v_0.copy()

        for n in range(maxiter):
            # Calculate x update step
            v_i = v + sigma*L(x)
            q_n = v_i - l2ball(v_i)

            # Calculate v update step
            temp = LT(2 * q_n - v)
            x_i = x - tau * sigma * temp
            p_n = l1(x_i, tau)

            # Update x and v
            x_new = x + llambda * (p_n - x)
            v_new = v + llambda * (q_n - v)

            # Get new norms
            norm2 = da.linalg.norm(x_new)
            norm1 = da.linalg.norm(x_new, 1)

            if norm1 == 0:
                norm1 = 1
            if norm2 == 0:
                norm2 = 1

            # get diff i.t.o. 1-norm
            diff1 = da.linalg.norm(x_new - x, 1) / norm1
            # get diff i.t.o. 2-norm
            diff2 = da.linalg.norm(x_new - x) / norm2

            if dask:
                x_new, v_new, norm1, norm2, diff1, diff2 = da.compute(x_new, v_new, norm1, norm2, diff1, diff2)

            print('L1 norm=', norm1, ' L2 norm=', norm2)
            print('Iter = %i, diff1 = %f, diff2 = %f' % (n, diff1, diff2))


            # Set new values to current values
            x = x_new
            v = v_new

            # import matplotlib.pyplot as plt
            # plt.figure("Primal Dual image")
            # plt.plot(x)
            # # plt.imshow(x.reshape([129, 129]))
            # plt.show()

            diff = max(diff1, diff2)
            if diff < tol:
                break

        return x_new

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

            # Get new norms
            norm2 = da.linalg.norm(x_new)
            norm1 = da.linalg.norm(x_new, 1)

            if norm1 == 0:
                norm1 = 1
            if norm2 == 0:
                norm2 = 1

            # get diff i.t.o. 1-norm
            diff1 = da.linalg.norm(x_new - x, 1) / norm1
            # get diff i.t.o. 2-norm
            diff2 = da.linalg.norm(x_new - x) / norm2

            if dask:
                x_new, v_new, norm1, norm2, diff1, diff2 = da.compute(x_new, v_new, norm1, norm2, diff1, diff2)

            print('L1 norm=', norm1, ' L2 norm=', norm2)
            print('Iter = %i, diff1 = %f, diff2 = %f' % (n, diff1, diff2))

            # Set new values to current values
            x = x_new
            v = v_new

            diff = max(diff1, diff2)
            if diff < tol:
                break

        return x_new

    return {
        'fbpd': lambda: fbpd(),
        'rpd': lambda: rescaled_pd(),
        'spd': lambda: symmetric_pd(),
        'rspd': lambda: rescaled_sym_pd(),
    }[solver]()

