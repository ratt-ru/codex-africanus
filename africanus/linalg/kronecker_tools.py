# -*- coding: utf-8 -*-


import numpy as np


def kron_matvec(A, b):
    """
    Computes the matrix vector product of
    a kronecker matrix in linear time.

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
    D = A.shape[0]
    N = b.size
    x = b
    for d in range(D):
        Gd = A[d].shape[0]
        X = np.reshape(x, (Gd, N//Gd))
        Z = np.einsum("ab,bc->ac", A[d], X)
        Z = np.einsum("ab -> ba", Z)
        x = Z.flatten()
    return x


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
    D = A.shape[0]
    L = np.zeros_like(A)
    for i in range(D):
        try:
            L[i] = np.linalg.cholesky(A[i])
        except Exception:  # add jitter
            L[i] = np.linalg.cholesky(A[i] + 1e-13*np.eye(A[i].shape[0]))
    return L
