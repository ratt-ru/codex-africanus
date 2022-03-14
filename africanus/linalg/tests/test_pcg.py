import numpy as np
np.random.seed(420)
from numpy.testing import assert_array_almost_equal
from africanus.linalg.pcg import pcg
from numba import njit

def test_pcg():
    N = 20
    x = np.random.randn(N) + 1.0j*np.random.randn(N)
    A = np.random.randn(100, N) + 1.0j*np.random.randn(100, N)
    A = A.conj().T @ A
    b = A.dot(x)

    Aop = njit(fastmath=True, inline='always')(lambda x: A.dot(x))
    M = np.diag(1.0/np.diag(A))
    Mop = njit(fastmath=True, inline='always')(lambda x: M.dot(x))
    res = pcg(Aop, b, np.zeros(b.size, dtype=b.dtype), M=Mop, tol=1e-6, maxit=N)

    # decimal is one less than that of tol
    assert_array_almost_equal(x, res, decimal=5)


if __name__=="__main__":
    test_pcg()