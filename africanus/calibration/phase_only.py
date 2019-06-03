# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numba
from africanus.calibration.utils import check_type
from africanus.averaging.support import unique_time
from africanus.util.docs import DocstringTemplate
from africanus.calibration.utils import residual_vis

@numba.jit(nopython=True, nogil=True, cache=True)
def jhj_and_jhr(time_indices, antenna1, antenna2, counts, 
                jones, residual, model, flag):
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
    jhr = np.zeros(jones.shape, dtype=np.complex128) 
    jhj = np.zeros(jones.shape, dtype=np.float64)
    
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
                                    tmp = sign * jones[t, p, nu, s, c] * \
                                                 model[s, row, nu, c] * \
                                                 np.conj(jones[t, q, nu, s, c])
                                    jhj[t, ant, nu, s, c] += (np.conj(tmp) * tmp).real
                                    jhr[t, ant, nu, s, c] += np.conj(tmp) * residual[row, nu, c]

    return jhj, jhr

#@numba.jit(nopython=True, nogil=True, cache=True)
def phase_only_GN(time_indices, antenna1, antenna2, 
                  counts, vis, flag, model, weight, 
                  jones=None, tol=1e-4, maxiter=100):
    # whiten data
    sqrtweights = np.sqrt(weight)
    vis *= sqrtweights
    model *= sqrtweights[None]
    
    # get dimensions
    n_dir, n_row, n_chan, n_cor = model.shape
    assert n_cor == 1 or n_cor == 2
    n_ant = np.maximum(antenna1.max(), antenna2.max()) + 1
    n_bl = n_ant * (n_ant - 1)//2
    
    # number of times
    n_tim = time_indices.size
    
    # set initial guess for the gains
    if jones is None:
        jones = np.ones((n_tim, n_ant, n_chan, n_dir, n_cor), dtype=np.complex128)

    eps = 1.0
    k = 0
    while eps > tol and k < maxiter:
        # keep track of old phases
        phases = np.angle(jones)
        
        # get residual
        residual = residual_vis(time_indices, antenna1, antenna2, counts, jones, vis, flag, model)

        # get diag(jhj) andf jhr
        jhj, jhr = jhj_and_jhr(model, residual, jones, antenna1, antenna2, time_indices, counts, flag)
        
        # implement update
        phases_new = phases + (jhr/jhj).real 
        jones = np.exp(1.0j * phases_new)

        # check convergence/iteration control
        eps = np.abs(phases_new - phases).max()
        k += 1
        # print("At iteration %i max diff = %f" % (k, eps))
    return jones, jhj, jhr, k

PHASE_CALIBRATION_DOCS = DocstringTemplate("""
Performs phase-only maximum likelihood 
calibration assuming scalar or diagonal 
inputs.

Parameters
----------
model : $(array_type)
    Model data values of shape :code:`(dir, row, chan, corr)`.
data : $(array_type)
    Data values of shape :code:`(row, chan, corr)`.
weight : $(array_type)
    Weight spectrum of shape :code:`(row, chan, corr)`.
    If the channel axis is missing weights are duplicated 
    for each channel.
antenna1 : $(array_type)
    First antenna indices of shape :code:`(row,)`.
antenna2 : $(array_type)
    Second antenna indices of shape :code:`(row,)`
time : $(array_type)
    Time values of shape :code:`(row,)`
flag : $(array_type)
    Flag data of shape :code:`(row, chan, corr)`.
tol : float, optional
    The tolerance of the solver. Defaults to 1e-4
maxiter: int, optional
    The maximum number of iterations. Defaults to 100.

Returns
-------
gains: $(array_type)
    Gain solutions of shape :code:`(time, ant, chan, dir, corr)`.
jhj: $(array_type)
    The diagonal of the Hessian of 
    shape :code:`(time, ant, chan, dir, corr)`.
jhr: $(array_type)
    Residuals projected into gain space 
    of shape :code:`(time, ant, chan, dir, corr)`.
k: int
    Number of iterations  
""")


try:
    phase_only_GN.__doc__ = PHASE_CALIBRATION_DOCS.substitute(
                                    array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass

JHJ_AND_JHR_DOCS = DocstringTemplate("""
Computes the diagonal of the Hessian and
the residual projected in to gain space. 
These are the terms required to perform
phase-only maximum likelihood calibration
assuming scalar or diagonal 
inputs.

Parameters
----------
model : $(array_type)
    Model data values of shape :code:`(dir, row, chan, corr)`.
data : $(array_type)
    Data values of shape :code:`(row, chan, corr)`.
weight : $(array_type)
    Weight spectrum of shape :code:`(row, chan, corr)`.
    If the channel axis is missing weights are duplicated 
    for each channel.
antenna1 : $(array_type)
    First antenna indices of shape :code:`(row,)`.
antenna2 : $(array_type)
    Second antenna indices of shape :code:`(row,)`
time : $(array_type)
    Time values of shape :code:`(row,)`
flag : $(array_type)
    Flag data of shape :code:`(row, chan, corr)`.
tol : float, optional
    The tolerance of the solver. Defaults to 1e-4
maxiter: int, optional
    The maximum number of iterations. Defaults to 100.

Returns
-------
gains: $(array_type)
    Gain solutions of shape :code:`(time, ant, chan, dir, corr)`.
jhj: $(array_type)
    The diagonal of the Hessian of 
    shape :code:`(time, ant, chan, dir, corr)`.
jhr: $(array_type)
    Residuals projected into gain space 
    of shape :code:`(time, ant, chan, dir, corr)`.
k: int
    Number of iterations  
""")


try:
    jhj_and_jhr.__doc__ = JHJ_AND_JHR_DOCS.substitute(
                                    array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass