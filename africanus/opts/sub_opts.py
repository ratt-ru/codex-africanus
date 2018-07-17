import numpy as np


def proj_l2ball(x, eps, y):
    # projection of x onto the l2 ball centered in y with radius eps
    p = x-y
    p = p * np.minimum(eps/np.linalg.norm(p), np.ones_like(y))
    p = p+y
    return p


def proj_l1_plus_pos(x, tau):
    return np.where(x < tau, 0.0, x-tau)


def pow_method(A, At, im_size, tol=1e-6, max_iter=200):
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
    x = np.random.randn(im_size[0], im_size[1])
    x /= np.linalg.norm(x, 'fro')
    init_val = 1

    for i in range(max_iter):
        y = A(x)
        x = At(y)
        val = np.linalg.norm(x, 'fro')
        rel_var = np.abs(val - init_val) / init_val
        if rel_var < tol:
            break
        init_val = val
        x /= val

    return val
