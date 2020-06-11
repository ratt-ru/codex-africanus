# -*- coding: utf-8 -*-


import numpy as np
from numba import njit

@njit(fastmath=True, inline='always')
def kron_N(x):
    """
    Computes N = N_1 x N_2 x ... x N_D i.e.
    the total number of rows in a kronecker matrix

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        An array of arrays holding matrices/vectors [x1, x2, ..., xD]

    Returns
    -------
    N : int
        The total number of rows in a kronecker matrix or vector
    """
    D = len(x)
    N = 1
    for i in range(D):
        N *= x[i].shape[0]
    return N

@njit(fastmath=True, inline='always')
def kron_matvec(A, b):
    """
    Computes the matrix vector product of
    a kronecker matrix in linear time.
    Assumes A consists of kronecker product
    of square matrices.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        An array of arrays holding
        matrices [K0, K1, ...] where
        :math:`A = K_0 \\otimes K_1 \\otimes \\cdots`
    b : :class:`numpy.ndarray`
        The right hand side vector

    Returns
    -------
    x : :class:`numpy.ndarray`
        The result of :code:`A.dot(b)`
    """
    D = len(A)
    N = b.size
    x = b
    for d in range(D):
        Gd = A[d].shape[0]
        X = np.reshape(x, (Gd, N//Gd))
        Z = np.zeros((Gd, N//Gd), dtype=A[0].dtype)
        for i in range(Gd):
            for j in range(N//Gd):
                for k in range(Gd):
                    Z[i, j] += A[d][i, k] * X[k, j]
        x = Z.T.flatten()
    return x

@njit(fastmath=True, inline='always')
def kron_tensorvec(A, b):
    """
    Matrix vector product of kronecker matrix A with
    vector b. A can be made up of an arbitrary kronecker
    product of non-square matrices.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        An array of arrays holding
        matrices [K0, K1, ...] where
        :math:`A = K_0 \\otimes K_1 \\otimes \\cdots`
    b : :class:`numpy.ndarray`
        The right hand side vector

    Returns
    -------
    x : :class:`numpy.ndarray`
        The result of :code:`A.dot(b)`
    """
    D = len(A)
    # get shape of sub-matrices
    G = np.zeros(D, dtype=np.int64)
    M = np.zeros(D, dtype=np.int64)
    for d in range(D):
        M[d], G[d] = A[d].shape
    x = b
    for d in range(D):
        Gd = G[d]
        rem = np.prod(np.delete(G, d))
        X = np.reshape(x, (Gd, rem))
        Z = np.zeros((A[d].shape[0], rem), dtype=A[0].dtype)
        for i in range(A[d].shape[0]):
            for j in range(rem):
                for k in range(A[d].shape[1]):
                    Z[i, j] += A[d][i, k] * X[k, j]
        x = Z.T.flatten()
        # replace with new dimension
        G[d] = M[d]
    return x

@njit(fastmath=True, inline='always')
def kron_matmat(A, B):
    """
    Computes the product between a kronecker matrix A
    and some RHS matrix B
    Parameters
    ----------
    A : :class:`numpy.ndarray`
        An array of arrays holding
        matrices [K0, K1, ...] where
        :math:`A = K_0 \\otimes K_1 \\otimes \\cdots`
    B : :class:`numpy.ndarray`
        The RHS matrix

    Returns
    -------
    x : :class:`numpy.ndarray`
        The result of :code:`A.dot(B)`
    """
    M = B.shape[1]
    N = kron_N(A)
    C = np.zeros((N, M), dtype=A[0].dtype)
    for i in range(M):
        tmp = np.ascontiguousarray(B[:, i])  # seems unavoidable
        C[:, i] = kron_matvec(A, tmp)
    return C

@njit(fastmath=True, inline='always')
def kron_tensormat(A, B):
    """
    Computes the matrix product between A kronecker matrix A
    and some RHS matrix B. Does not assume A to consist of a
    kronecker product of square matrices.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        An array of arrays holding
        matrices [K0, K1, ...] where
        :math:`A = K_0 \\otimes K_1 \\otimes \\cdots`
    B : :class:`numpy.ndarray`
        The RHS matrix

    Returns
    -------
    x : :class:`numpy.ndarray`
        The result of :code:`A.dot(B)`
    """
    M = B.shape[1]  # the product of Np_1 x Np_2 x ... x Np_3

    N = kron_N(A)
    C = np.zeros((N, M), dtype=A[0].dtype)
    for i in range(M):
        tmp = np.ascontiguousarray(B[:, i])  # seems unavoidable
        C[:, i] = kron_tensorvec(A, tmp)
    return C


def kron_cholesky(A):
    """
    Computes the Cholesky decomposition
    of a kronecker matrix as a kronecker
    matrix of Cholesky factors.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        An array of arrays holding
        matrices [K0, K1, ...] where
        :math:`A = K_0 \\otimes K_1 \\otimes \\cdots`

    Returns
    -------
    L : :class:`numpy.ndarray`
        An array of arrays holding
        matrices [L0, L1, ...] where
        :math:`L = L_0 \\otimes L_1 \\otimes \\cdots`
        and each :code:`Li = cholesky(Ki)`
    """
    D = len(A)
    L = ()
    for i in range(D):
        try:
            L += (np.linalg.cholesky(A[i]),)
        except Exception:  # add jitter
            L += (np.linalg.cholesky(A[i] + 1e-13*np.eye(A[i].shape[0])),)
    return L


def kron_inv(A):
    D = len(A)
    Kinv = ()
    for i in range(D):
        Kinv += (np.linalg.inv(A[i] + 1e-13*np.eye(A[i].shape[0])),)
    return Kinv