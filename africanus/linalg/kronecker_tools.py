# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def kron_matvec(A, b):
    """
    Computes matrix vector product of kronecker matrix in linear time. 
    :param A: an array of arrays holding matrices [K0, K1, K2, ...]
    :param b: the RHS vector
    :return: A.dot(b)
    """
    D = A.shape[0]
    N = b.size
    x = b
    for d in xrange(D):
        Gd = A[d].shape[0]
        X = np.reshape(x,(Gd, N//Gd))
        Z = np.einsum("ab,bc->ac", A[d], X)
        Z = np.einsum("ab -> ba", Z)
        x = Z.flatten()
    return x

def kron_cholesky(A):
    """
    Computes the cholesky decomposition of a kronecker matrix
    :param A: an array of arrays holding matrices [K1, K2, K3, ...]
    :return: 
    """
    D = A.shape[0]
    L = np.zeros_like(A)
    for i in xrange(D):
        try:
            L[i] = np.linalg.cholesky(A[i])
        except: # add jitter
            L[i] = np.linalg.cholesky(A[i] + 1e-13*np.eye(A[i].shape[0]))
    return L