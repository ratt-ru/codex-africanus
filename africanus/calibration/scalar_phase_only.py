# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from africanus.util.numba import generated_jit, njit
from africanus.averaging.support import unique_time

@njit(nopython=True, nogil=True, cache=True)
def give_residual(model, data, gains, antenna1, antenna2, time_indices, counts, flag):
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
    time_indices- (n_tim) int array containing change points for unique time stamps
    counts      - (n_tim) int array containing counts of unique time in time
    flag        - (n_row, n_fre, n_cor) bool array containing flags
    
    Output:
    residual    - (n_row, n_fre, n_cor) complex array containing residual
    """
    # get dimensions
    n_dir, n_row, n_chan, n_cor = model.shape
    assert n_cor == 1 or n_cor == 2
    n_ant = np.maximum(antenna1.max(), antenna2.max()) + 1
    n_bl = n_ant * (n_ant - 1)//2
    
    # get number of times
    n_tim = time_indices.size

    # Assume this for now
    assert n_row == n_tim * n_bl

    # so we just need to subtract model
    residual = data.copy()
    
    # time iterator
    for t in range(n_tim):
        # I = np.arange(time_indices[t], time_indices[t] + counts[t])
        # anta = antenna1[I]
        # antb = antenna2[I]
        # baseline iterator
        for row in range(time_indices[t], time_indices[t] + counts[t]):
            p = int(antenna1[row])
            q = int(antenna2[row])
            gp = gains[t, p]
            gqH = np.conj(gains[t, q])
            for nu in range(n_chan):
                for s in range(n_dir):
                    for c in range(n_cor):
                        if not flag[row, nu, c]:
                            residual[row, nu, c] -= gp[nu, s, c] * model[s, row, nu, c] * gqH[nu, s, c]

    return residual

@njit(nopython=True, nogil=True, cache=True)
def give_jhj_and_jhr(model, residual, gains, antenna1, antenna2, time_indices, counts, flag):
    """
    Computes the jhj and jhr required for phase only calibration assuming scalar
    gains and that off diagonal correlations (i.e. xy, yx or rl, lr) are zero.
    Only computes the diagonal of jhj.
    Inputs:
    model       - (n_dir, n_row, n_fre, n_cor, n_cor) complex array containg model data 
                  for all directions
    residual    - (n_row, n_fre, n_cor) complex array containg residual
    gains       - (n_tim, n_ant, n_fre, n_dir, n_cor) complex array containing current estimate of gains
    antenna1    - (n_row) int array containing antenna1 column
    antenna2    - (n_row) int array containing antenna2 column
    time_indices- (n_tim) int array containing change points for unique time stamps
    counts      - (n_tim) int array containing counts of unique time in time
    flag        - (n_row, n_fre, n_cor) bool array containing flags
    """
    # get dimensions
    n_dir, n_row, n_chan, n_cor = model.shape
    assert n_cor == 1 or n_cor == 2
    n_ant = np.maximum(antenna1.max(), antenna2.max()) + 1
    n_bl = n_ant * (n_ant - 1)//2
    
    # get number of times
    n_tim = time_indices.size

    # Assume this for now
    assert n_row == n_tim * n_bl

    # initialise
    jhr = np.zeros(gains.shape, dtype=np.complex128) 
    jhj = np.zeros(gains.shape, dtype=np.float64)
    
    for t in range(n_tim):
        I = np.arange(time_indices[t], time_indices[t] + counts[t])
        for ant in range(n_ant):
            # find where either antenna == ant
            # these will be mutually exclusive since no autocorrs
            for row in I:
                if antenna1[row] == ant or antenna2[row] == ant:
                    p = antenna1[row]
                    q = antenna2[row]
                    for nu in range(n_chan):
                        for s in range(n_dir):
                            for c in range(n_cor):
                                if not flag[row, nu, c]:
                                    if ant == p:
                                        sign = 1.0j
                                    elif ant == q:
                                        sign = -1.0j
                                    else:
                                        print("Something has gone wrong")               
                                    tmp = sign * gains[t, p, nu, s, c] * model[s, row, nu, c] * np.conj(gains[t, q, nu, s, c])
                                    jhj[t, ant, nu, s, c] += (np.conj(tmp) * tmp).real
                                    jhr[t, ant, nu, s, c] += np.conj(tmp) * residual[row, nu, c]

    return jhj, jhr

@njit(nopython=True, nogil=True, cache=True)
def phase_only_calibration(model, data, weight, antenna1, antenna2, time, flag,
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
    flag       - (n_row, n_fre, n_cor) bool array containing flags
    """
    # # whiten data
    # sqrtweights = np.sqrt(weight)
    # data *= sqrtweights
    # model *= sqrtweights[None]
    
   # get dimensions
    n_dir, n_row, n_chan, n_cor = model.shape
    assert n_cor == 1 or n_cor == 2
    n_ant = np.maximum(antenna1.max(), antenna2.max()) + 1
    n_bl = n_ant * (n_ant - 1)//2
    
    # get unique times
    tmp = unique_time(time)
    time_indices = tmp[1]
    counts = tmp[3]
    #_, time_indices, _, counts = 
    n_tim = time_indices.size
    
    # set initial guess for the gains
    gains = np.ones((n_tim, n_ant, n_chan, n_dir, n_cor), dtype=np.complex128)

    eps = 1.0
    k = 0
    while eps > tol and k < maxiter:
        # keep track of old phases
        phases = np.angle(gains)
        
        # get residual
        residual = give_residual(model, data, gains, antenna1, antenna2, time_indices, counts, flag)

        # get diag(jhj) andf jhr
        jhj, jhr = give_jhj_and_jhr(model, residual, gains, antenna1, antenna2, time_indices, counts, flag)
        
        # implement update
        phases_new = phases + (jhr/jhj).real 
        gains = np.exp(1.0j * phases_new)

        # check convergence/iteration control
        eps = np.abs(phases_new - phases).max()
        k += 1
        # print("At iteration %i max diff = %f" % (k, eps))
    return gains, jhj, jhr