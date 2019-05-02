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
    
    Output:
    jac         - (n_row * n_fre * n_cor, n_tim * n_ant * n_fre * n_dir * n_cor) complex array containing
                  Jacobian
    residual    - (n_row * n_fre * n_cor) complex array containing residual
    NOTE - mainly for testing!!
    """
    # get dimensions
    n_row, n_chan, n_cor = data.shape
    assert n_cor == 1 # or n_cor == 2
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
                        # there are only going to be two non-zero terms. 
                        # One for p 
                        jac[t * n_bl + i, nu, c, t, p, nu, s, c] += 1.0j*tmp
                        # and one for q
                        jac[t * n_bl + i, nu, c, t, q, nu, s, c] += -1.0j*tmp

    return jac.reshape(np.prod(data.shape), np.prod(gains.shape)), residual.reshape(np.prod(data.shape))

def give_residual(model, data, gains, antenna1, antenna2, time, flags):
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
    
    Output:
    residual    - (n_row * n_fre * n_cor) complex array containing residual
    NOTE - mainly for testing!!
    """
    # get dimensions
    n_dir, n_row, n_chan, n_cor = model.shape
    assert n_cor == 1 # or n_cor == 2
    n_ant = np.maximum(antenna1.max(), antenna2.max()) + 1
    n_bl = n_ant * (n_ant - 1)//2
    
    # get unique times
    unique_times = np.unique(time)
    n_tim = unique_times.size

    # Assume this for now
    assert n_row == n_tim * n_bl

    # so we just need to subtract model
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
                        residual[t * n_bl + i, nu, c] -= gp[nu, s, c] * model[s, t * n_bl + i, nu, c] * gqH[nu, s, c]

    return residual

def give_jhj_and_jhr(model, residual, gains, antenna1, antenna2, time, flags):
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
    time        - (n_row) float array containing observation time stamps
    flags       - (n_row, n_fre, n_cor) bool array containing flags
    """
    # get dimensions
    n_dir, n_row, n_chan, n_cor = model.shape
    assert n_cor == 1 # or n_cor == 2
    n_ant = np.maximum(antenna1.max(), antenna2.max()) + 1
    n_bl = n_ant * (n_ant - 1)//2
    
    # get unique times
    unique_times = np.unique(time)
    n_tim = unique_times.size

    # Assume this for now
    assert n_row == n_tim * n_bl

    # initialise
    jhr = np.zeros(gains.shape, dtype=np.complex128) 
    jhj = np.zeros(gains.shape, dtype=np.float64)
    
    for t in range(n_tim):
        I = np.argwhere(time == unique_times[t]).squeeze()
        for ant in range(n_ant):
            # find where either antenna == ant
            # these will be mutually exclusive since no autocorrs
            Ip = np.argwhere(antenna1[I] == ant).squeeze()
            Iq = np.argwhere(antenna2[I] == ant).squeeze()
            # get corresponding rows
            rowsp = I[Ip]
            rowsq = I[Iq]
            # stack it up
            Ipq1 = np.vstack((antenna1[I][Ip], antenna2[I][Ip], rowsp)).T
            Ipq2 = np.vstack((antenna1[I][Iq], antenna2[I][Iq], rowsq)).T
            Ipq = np.vstack((Ipq1, Ipq2))
            for p, q, row in Ipq:
                for nu in range(n_chan):
                    for s in range(n_dir):
                        for c in range(n_cor):
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
    unique_times = np.unique(time)
    n_tim = unique_times.size
    
    # set initial guess for the gains
    gains = np.ones((n_tim, n_ant, n_chan, n_dir, n_cor), dtype=np.complex128)

    eps = 1.0
    k = 0
    while eps > tol and k < maxiter:
        # keep track of old phases
        phases = np.angle(gains)
        
        # get residual
        residual = give_residual(model, data, gains, antenna1, antenna2, time, flags)

        # get diag(jhj) andf jhr
        jhj, jhr = give_jhj_and_jhr(model, residual, gains, antenna1, antenna2, time, flags)
        
        # implement update
        phases_new = phases + (jhr/jhj).real 
        gains = np.exp(1.0j * phases_new)

        # check convergence/iteration control
        eps = np.abs(phases_new - phases).max()
        k += 1
        print("At iteration %i max diff = %f" % (k, eps))
    return gains, jhj, jhr