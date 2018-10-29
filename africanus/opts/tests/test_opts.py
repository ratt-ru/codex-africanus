#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

import dask.array as da
import numpy as np

import pytest


def test_power_method():
    """
    test the operation of the power method which approximates the largest eigenvalue of the matrix,
    both methods should result in the same value.
    """
    from africanus.opts.sub_opts import pow_method as pm

    eig = 0

    while eig is 0:
        A = np.random.randn(10, 10)
        G = A.T.dot(A)
        eig_vals = np.linalg.eigvals(G)
        if min(eig_vals) > 0:
            eig = max(eig_vals)

    spec = pm(G.dot, G.conj().T.dot, [10, 1])

    assert (abs(eig-spec) < 1e-8)
