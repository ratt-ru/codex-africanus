import numpy as np
from numba import njit

@njit(fastmath=True, inline='always')
def pcg(A, b, x0, M, tol=1e-7, maxit=500):
    """
    Masked preconditioned conjugate gradient algorithm to solve problems of the form

    Ax = b

    where b is a vector that possibly contains masked or invalid entries
    and A is a positive definite and Hermitian matrix.
    """
    r = A(x0) - b
    y = M(r)
    p = -y
    rnorm = np.real(np.vdot(r, y))
    if np.isnan(rnorm) or rnorm == 0.0:
        eps0 = 1.0
    else:
        eps0 = rnorm
    k = 0
    x = x0
    while rnorm/eps0 > tol**2 and k < maxit:
        xp = x.copy()
        rp = r.copy()
        Ap = A(p)
        rnorm = np.real(np.vdot(r, y))
        alpha = rnorm/np.vdot(p, Ap)
        x = xp + alpha*p
        r = rp + alpha*Ap
        y = M(r)
        rnorm_next = np.real(np.vdot(r, y))
        beta = rnorm_next/rnorm
        p = beta*p - y
        rnorm = rnorm_next
        k += 1

    return x