import numpy as np
np.random.seed(420)
from numpy.testing import assert_array_almost_equal
from africanus.linalg.kronecker_tools import kron_matvec, kron_tensorvec, kron_matmat, kron_tensormat


def test_matvec():
    # Construct three square matrices and compare
    # kron_matvec to explicit matrix vector product
    x = np.random.randn(3, 3)
    y = np.random.randn(4, 4)
    z = np.random.randn(5, 5)

    K1 = (x, y, z)
    K2 = np.kron(x, np.kron(y, z))

    tmp = np.random.randn(3*4*5)

    res1 = kron_matvec(K1, tmp)
    res2 = K2.dot(tmp)

    assert_array_almost_equal(res1, res2, decimal=13)

def test_tensorvec():
    # Construct three non-square matrices and compare
    # kron_tensorvec to explicit matrix vector product
    x = np.random.randn(3, 4)
    y = np.random.randn(4, 5)
    z = np.random.randn(5, 6)

    K1 = (x, y, z)
    K2 = np.kron(x, np.kron(y, z))

    tmp = np.random.randn(4*5*6)

    res1 = kron_tensorvec(K1, tmp)
    res2 = K2.dot(tmp)

    assert_array_almost_equal(res1, res2, decimal=13)

def test_matmat():
    # Construct three square matrices and compare
    # kron_matvec to explicit matrix matrix product
    x = np.random.randn(3, 3)
    y = np.random.randn(4, 4)
    z = np.random.randn(5, 5)

    K1 = (x, y, z)
    K2 = np.kron(x, np.kron(y, z))

    tmp = np.random.randn(3*4*5, 10)

    res1 = kron_matmat(K1, tmp)
    res2 = K2.dot(tmp)

    assert_array_almost_equal(res1, res2, decimal=13)

def test_tensormat():
    # Construct three non-square matrices and compare
    # kron_tensorvec to explicit matrix matrix product
    x = np.random.randn(3, 4)
    y = np.random.randn(4, 5)
    z = np.random.randn(5, 6)

    K1 = (x, y, z)
    K2 = np.kron(x, np.kron(y, z))

    tmp = np.random.randn(4*5*6, 10)

    res1 = kron_tensormat(K1, tmp)
    res2 = K2.dot(tmp)

    assert_array_almost_equal(res1, res2, decimal=13)

if __name__=="__main__":
    test_matvec()
    test_tensorvec()
    test_matmat()
    test_tensormat()
