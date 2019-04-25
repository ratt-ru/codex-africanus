#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def give_jacobian(model, data, gains, antenna1, antenna2, time, flags):
    """
    Computes the Jacobian required for phase only calibration assuming scalar
    gains and that off diagonal correlations (i.e. xy, yx or rl, lr) are zero.
    Inputs:
    model       - (n_dir, n_row, n_fre, n_cor, n_cor) complex array containg model data 
                  for all directions
    data        - (n_row, n_fre, n_cor) complex array containg data
    gains       - (n_tim, n_ant, n_fre, n_dir, n_cor) complex array containing current estimate of gains
    antenna1    - (n_row) int array containing antenna1 column
    antenna2    - (n_row) int array containing antenna2 column
    time        - (n_row) float array containing observation time stamps
    flags       - (n_row, n_fre, n_cor) bool array containing flags
    """
    # get dimensions
    n_row, n_chan, n_cor = data.shape
    assert n_cor == 1 or n_cor == 2
    n_dir, _, _, _ = model.shape
    n_ant = np.maximum(antenna1.max(), antenna2.max()) + 1
    n_bl = n_ant * (n_ant - 1)//2
    
    # get unique times
    unique_times = np.unique(time)
    n_tim = unique_times.size

    # Assume this for now
    assert n_row == n_tim * n_bl

    # the Jacobian is always the shape of the data by the shape of
    # the parameter vector 
    jac = np.zeros(data.shape + gains.shape, dtype=np.complex128)
    residual = data.copy()
    
    # time iterator
    for t in range(n_tim):
        I = np.argwhere(time == unique_times[t])
        anta = antenna1[I]
        antb = antenna2[I]
        # baseline iterator
        for i in range(I.size):
            assert I.size == n_bl
            p = int(anta[i])
            q = int(antb[i])
            gp = gains[t, p]
            gqH = np.conj(gains[t, q])
            for nu in range(n_chan):
                for s in range(n_dir):
                    for c in range(n_cor):
                        tmp = (gp[nu, s, c] * model[s, t * n_bl + i, nu, c] * gqH[nu, s, c])
                        residual[t * n_bl + i, nu, c] -= tmp
                        jac[t * n_bl + i, nu, c, t, p, nu, s, c] = 1.0j*tmp
                        jac[t * n_bl + i, nu, c, t, q, nu, s, c] = -1.0j*tmp

    return jac.reshape(np.prod(data.shape), np.prod(gains.shape)), residual.reshape(np.prod(data.shape))

def phase_only_calibration(model, data, weight, antenna1, antenna2, time, flags,
                           tol=1e-4, maxiter=100):
    """
    Inputs:
    model       - (n_dir, n_row, n_fre, n_cor, n_cor) complex array containg model data 
                  for all directions
    data        - (n_row, n_fre, n_cor) complex array containg data
    weight      - either (n_row, n_fre, n_cor) or (n_row, ncor) float array containing data weights
    antenna1    - (n_row) int array containing antenna1 column
    antenna2    - (n_row) int array containing antenna2 column
    time        - (n_row) float array containing observation time stamps
    flags       - (n_row, n_fre, n_cor) bool array containing flags
    """
    # whiten data
    sqrtweights = np.sqrt(weight)
    data *= sqrtweights
    model *= sqrtweights[None]
    
   # get dimensions
    n_row, n_chan, n_cor = data.shape
    assert n_cor == 1 or n_cor == 2
    n_dir, _, _, _ = model.shape
    n_ant = np.maximum(antenna1.max(), antenna2.max()) + 1
    n_bl = n_ant * (n_ant - 1)//2
    
    # get unique times
    unique_times = np.unique(time)
    n_tim = unique_times.size
    
    # set initial guess for the gains
    gains = np.ones((n_tim, n_ant, n_chan, n_dir, n_cor), dtype=np.complex128)

    eps = 1.0
    k = 0
    while eps > tol and k < maxiter:
        # keep track of old phases
        phases = np.angle(gains).flatten()
        
        # get Jacobian and residual
        jac, residual = give_jacobian(model, data, gains, antenna1, antenna2, time, flags)
        
        # compute J.H J (will be diagonal so only need to compute that part)
        # JHJ = Jac.conj().T.dot(Jac)
        jhj = np.einsum('ij,ji->i', jac.conj().T, jac)

        # compute JHr
        jhr = jac.conj().T.dot(residual)
        
        # implement update
        phases_new = phases + (jhr/jhj).real 
        
        # plt.figure('phases')
        # for ant in range(Na):
        #     plt.plot(phases_new.reshape(Ntime, Na)[:, ant])
        # plt.show()
        
        gains = np.exp(1.0j * phases_new.reshape(n_tim, n_ant, n_chan, n_dir, n_cor))
        eps = np.abs(phases_new - phases).max()
        k += 1
        print("At iteration %i max diff = %f" % (k, eps))
    return gains, jac, residual