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


def power_dask(L, LT, im_size, tol=1e-8, max_iter=2000):
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


def da_get_diff(x_new, x, n):

    # Get new norms
    norm2 = da.linalg.norm(x_new).compute()
    norm1 = da.linalg.norm(x_new, 1).compute()

    if norm1 == 0:
        norm1 = 1
    if norm2 == 0:
        norm2 = 1

    # get diff i.t.o. 1-norm
    diff1 = da.linalg.norm(x_new - x, 1).compute() / norm1
    # get diff i.t.o. 2-norm
    diff2 = da.linalg.norm(x_new - x).compute() / norm2

    print('L1 norm=', norm1, ' L2 norm=', norm2)
    print('Iter = %i, diff1 = %f, diff2 = %f' % (n, diff1, diff2))

    return da.maximum(diff1, diff2)


def get_diff(x_new, x, n):

    # Get new norms
    norm2 = np.linalg.norm(x_new)
    norm1 = np.linalg.norm(x_new, 1)

    if norm1 == 0:
        norm1 = 1
    if norm2 == 0:
        norm2 = 1

    # get diff i.t.o. 1-norm
    diff1 = np.linalg.norm(x_new - x, 1) / norm1
    # get diff i.t.o. 2-norm
    diff2 = np.linalg.norm(x_new - x) / norm2

    print('L1 norm=', norm1, ' L2 norm=', norm2)
    print('Iter = %i, diff1 = %f, diff2 = %f' % (n, diff1, diff2))

    return np.maximum(diff1, diff2)

