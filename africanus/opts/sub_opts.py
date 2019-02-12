import numpy as np
import dask.array as da


def proj_l2ball(x, eps, y):
    # projection of x onto the l2 ball centered in y with radius eps
    p = x-y
    p = p * np.minimum(eps/np.linalg.norm(p), np.ones_like(y))
    p = p+y
    return p


def da_proj_l2ball(x, eps, y):
    # projection of x onto the l2 ball centered in y with radius eps
    p = x-y
    p = p * da.minimum(eps/da.linalg.norm(p), da.ones_like(y))
    p = p+y
    return p


def proj_l1_plus_pos(x, tau):
    return np.where(x < tau, 0.0, x-tau)


def da_proj_l1_plus_pos(x, tau):
    return da.where(x < tau, 0.0, x-tau)


def power_dask(L, LT, im_size, tol=1e-12, max_iter=2000):
    np.random.seed(123)
    x = da.random.random(im_size, chunks=im_size)
    x /= da.linalg.norm(x, 'fro')
    init_val = 1

    for i in range(max_iter):
        y = L(x)
        x = LT(y)
        val = da.linalg.norm(x, 'fro')
        rel_var = np.abs(val - init_val) / init_val
        if rel_var < tol:
            break
        if i % 10 == 0:
            print("Iter {0}: {1}".format(i, rel_var))
        init_val = val
        x /= val
    print('Spectral norm=', np.sqrt(val))
    return np.sqrt(val)


def pow_method(L, LT, im_size, tol=1e-12, max_iter=2000):
    """
    @author: mjiang
    ming.jiang@epfl.ch
    Computes the spectral radius (maximum eigen value) of the operator A

    @param A: function handle of direct operator

    @param At: function handle of adjoint operator

    @param im_size: size of the image

    @param tol: tolerance of the error, stopping criterion

    @param max_iter: max iteration

    @return: spectral radius of the operator
    """
    np.random.seed(123)
    x = np.random.randn(*im_size)
    x /= np.linalg.norm(x, 'fro')
    init_val = 1

    for i in range(max_iter):
        y = L(x)
        x = LT(y)
        val = np.linalg.norm(x, 'fro')
        rel_var = np.abs(val - init_val) / init_val
        if rel_var < tol:
            break
        if i % 10 == 0:
            print("Iter {0}: {1}".format(i, rel_var))
        init_val = val
        x /= val
    print('Spectral norm=', np.sqrt(val))
    return np.sqrt(val)

