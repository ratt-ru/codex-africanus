# -*- coding: utf-8 -*-

import numpy as np
from africanus.util.docs import DocstringTemplate
from africanus.calibration.utils import residual_vis, check_type
from africanus.util.numba import generated_jit, njit
from africanus.calibration.utils.utils import DIAG_DIAG, DIAG, FULL


def jacobian_factory(mode):
    if mode == DIAG_DIAG:
        def jacobian(a1j, blj, a2j, sign, out):
            out[...] = sign * a1j * blj * a2j.conjugate()
            # for c in range(out.shape[-1]):
            #     out[c] = sign * a1j[c] * blj[c] * a2j[c].conjugate()
    elif mode == DIAG:
        def jacobian(a1j, blj, a2j, sign, out):
            out[...] = 0
    elif mode == FULL:
        def jacobian(a1j, blj, a2j, sign, out):
            out[...] = 0
    return njit(nogil=True, inline='always')(jacobian)


@generated_jit(nopython=True, nogil=True, cache=True, fastmath=True)
def compute_jhj_and_jhr(time_bin_indices, time_bin_counts, antenna1,
                        antenna2, jones, residual, model, flag):

    mode = check_type(jones, residual)
    if mode != DIAG_DIAG:
        raise NotImplementedError("Only DIAG-DIAG case has been implemented")

    jacobian = jacobian_factory(mode)

    def _jhj_and_jhr_fn(time_bin_indices, time_bin_counts, antenna1,
                        antenna2, jones, residual, model, flag):
        # for chunked dask arrays we need to adjust the chunks to
        # start counting from zero (see also map_blocks)
        time_bin_indices -= time_bin_indices.min()
        jones_shape = np.shape(jones)
        n_tim = jones_shape[0]
        n_chan = jones_shape[2]
        n_dir = jones_shape[3]

        # storage arrays
        jhr = np.zeros(jones.shape, dtype=jones.dtype)
        jhj = np.zeros(jones.shape, dtype=jones.real.dtype)
        # tmp array the shape of jones_corr
        jac = np.zeros_like(jones[0, 0, 0, 0], dtype=jones.dtype)
        for t in range(n_tim):
            for row in range(time_bin_indices[t],
                             time_bin_indices[t] + time_bin_counts[t]):
                p = antenna1[row]
                q = antenna2[row]
                for nu in range(n_chan):
                    if np.any(flag[row, nu]):
                        continue
                    gp = jones[t, p, nu]
                    gq = jones[t, q, nu]
                    for s in range(n_dir):
                        # for the derivative w.r.t. antenna p
                        jacobian(gp[s], model[row, nu, s], gq[s], 1.0j, jac)
                        jhj[t, p, nu, s] += (np.conj(jac) * jac).real
                        jhr[t, p, nu, s] += (np.conj(jac) * residual[row, nu])
                        # for the derivative w.r.t. antenna q
                        jacobian(gp[s], model[row, nu, s], gq[s], -1.0j, jac)
                        jhj[t, q, nu, s] += (np.conj(jac) * jac).real
                        jhr[t, q, nu, s] += (np.conj(jac) * residual[row, nu])
        return jhj, jhr
    return _jhj_and_jhr_fn


@generated_jit(nopython=True, nogil=True, cache=True, fastmath=True)
def compute_jhj(time_bin_indices, time_bin_counts, antenna1,
                antenna2, jones, model, flag):

    mode = check_type(jones, model, vis_type='model')

    jacobian = jacobian_factory(mode)

    def _compute_jhj_fn(time_bin_indices, time_bin_counts, antenna1,
                        antenna2, jones, model, flag):
        # for dask arrays we need to adjust the chunks to
        # start counting from zero
        time_bin_indices -= time_bin_indices.min()
        jones_shape = np.shape(jones)
        n_tim = jones_shape[0]
        n_chan = jones_shape[2]
        n_dir = jones_shape[3]

        jhj = np.zeros(jones.shape, dtype=jones.real.dtype)
        # tmp array the shape of jones_corr
        jac = np.zeros_like(jones[0, 0, 0, 0], dtype=jones.dtype)
        for t in range(n_tim):
            for row in range(time_bin_indices[t],
                             time_bin_indices[t] + time_bin_counts[t]):
                p = antenna1[row]
                q = antenna2[row]
                for nu in range(n_chan):
                    if np.any(flag[row, nu]):
                        continue
                    gp = jones[t, p, nu]
                    gq = jones[t, q, nu]
                    for s in range(n_dir):
                        jacobian(gp[s], model[row, nu, s], gq[s], 1.0j, jac)
                        jhj[t, p, nu, s] += (jac.conjugate() * jac).real
                        jacobian(gp[s], model[row, nu, s], gq[s], -1.0j, jac)
                        jhj[t, q, nu, s] += (jac.conjugate() * jac).real
        return jhj
    return _compute_jhj_fn


@generated_jit(nopython=True, nogil=True, cache=True, fastmath=True)
def compute_jhr(time_bin_indices, time_bin_counts, antenna1,
                antenna2, jones, residual, model, flag):

    mode = check_type(jones, model, vis_type='model')

    jacobian = jacobian_factory(mode)

    def _compute_jhr_fn(time_bin_indices, time_bin_counts, antenna1,
                        antenna2, jones, residual, model, flag):
        # for dask arrays we need to adjust the chunks to
        # start counting from zero
        time_bin_indices -= time_bin_indices.min()
        jones_shape = np.shape(jones)
        n_tim = jones_shape[0]
        n_chan = jones_shape[2]
        n_dir = jones_shape[3]

        jhr = np.zeros(jones.shape, dtype=jones.dtype)
        # tmp array the shape of jones_corr
        jac = np.zeros_like(jones[0, 0, 0, 0], dtype=jones.dtype)
        for t in range(n_tim):
            for row in range(time_bin_indices[t],
                             time_bin_indices[t] + time_bin_counts[t]):
                p = antenna1[row]
                q = antenna2[row]
                for nu in range(n_chan):
                    if np.any(flag[row, nu]):
                        continue
                    gp = jones[t, p, nu]
                    gq = jones[t, q, nu]
                    for s in range(n_dir):
                        jacobian(gp[s], model[row, nu, s], gq[s], 1.0j, jac)
                        jhr[t, p, nu, s] += jac.conjugate() * residual[row, nu]
                        jacobian(gp[s], model[row, nu, s], gq[s], -1.0j, jac)
                        jhr[t, q, nu, s] += jac.conjugate() * residual[row, nu]
        return jhr
    return _compute_jhr_fn

# LB - TODO somehow this generated_jit causes tests to fail
# @generated_jit(nopython=True, nogil=True, cache=True, fastmath=True)


def gauss_newton(time_bin_indices, time_bin_counts, antenna1,
                 antenna2, jones, vis, flag, model,
                 weight, tol=1e-4, maxiter=100):

    # whiten data
    sqrtweights = np.sqrt(weight)
    vis *= sqrtweights
    model *= sqrtweights[:, :, None]

    mode = check_type(jones, vis)

    # can avoid recomputing JHJ in DIAG_DIAG mode
    if mode == DIAG_DIAG:
        jhj = compute_jhj(time_bin_indices, time_bin_counts,
                          antenna1, antenna2, jones, model, flag)
    else:
        raise NotImplementedError("Only DIAG_DIAG mode implemented")

    eps = 1.0
    k = 0
    while eps > tol and k < maxiter:
        # keep track of old phases
        phases = np.angle(jones)

        # get residual TODO - we can avoid this in DIE case
        residual = residual_vis(time_bin_indices, time_bin_counts, antenna1,
                                antenna2, jones, vis, flag, model)

        jhr = compute_jhr(time_bin_indices, time_bin_counts,
                          antenna1, antenna2,
                          jones, residual, model, flag)

        # implement update
        phases_new = phases + 0.5 * (jhr/jhj).real
        jones = np.exp(1.0j * phases_new)

        # check convergence/iteration control
        eps = np.abs(phases_new - phases).max()
        k += 1

    return jones, jhj, jhr, k


GAUSS_NEWTON_DOCS = DocstringTemplate("""
Performs phase-only maximum likelihood
calibration using a Gauss-Newton optimisation
algorithm. Currently only DIAG mode is supported.

Parameters
----------
time_bin_indices : $(array_type)
    The start indices of the time bins
    of shape :code:`(utime)`
time_bin_counts : $(array_type)
    The counts of unique time in each
    time bin of shape :code:`(utime)`
antenna1 : $(array_type)
    First antenna indices of shape :code:`(row,)`.
antenna2 : $(array_type)
    Second antenna indices of shape :code:`(row,)`.
jones : $(array_type)
    Gain solutions of shape :code:`(time, ant, chan, dir, corr)`
    or :code:`(time, ant, chan, dir, corr, corr)`.
vis : $(array_type)
    Data values of shape :code:`(row, chan, corr)`
    or :code:`(row, chan, corr, corr)`.
flag : $(array_type)
    Flag data of shape :code:`(row, chan, corr)`
    or :code:`(row, chan, corr, corr)`.
model : $(array_type)
    Model data values of shape :code:`(row, chan, dir, corr)`
    or :code:`(row, chan, dir, corr, corr)`.
weight : $(array_type)
    Weight spectrum of shape :code:`(row, chan, corr)`.
    If the channel axis is missing weights are duplicated
    for each channel.
tol: float, optional
    The tolerance of the solver. Defaults to 1e-4.
maxiter: int, optional
    The maximum number of iterations. Defaults to 100.

Returns
-------
gains : $(array_type)
    Gain solutions of shape :code:`(time, ant, chan, dir, corr)`
    or shape :code:`(time, ant, chan, dir, corr, corr)`
jhj : $(array_type)
    The diagonal of the Hessian of shape
    :code:`(time, ant, chan, dir, corr)` or shape
    :code:`(time, ant, chan, dir, corr, corr)`
jhr : $(array_type)
    Residuals projected into gain space
    of shape :code:`(time, ant, chan, dir, corr)`
    or shape :code:`(time, ant, chan, dir, corr, corr)`.
k: int
    Number of iterations (will equal maxiter if
    not converged)
""")


try:
    gauss_newton.__doc__ = GAUSS_NEWTON_DOCS.substitute(
                            array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass

JHJ_AND_JHR_DOCS = DocstringTemplate("""
Computes the diagonal of the Hessian and
the residual locally projected in to gain space.

Parameters
----------
time_bin_indices : $(array_type)
    The start indices of the time bins
    of shape :code:`(utime)`
time_bin_counts : $(array_type)
    The counts of unique time in each
    time bin of shape :code:`(utime)`
antenna1 : $(array_type)
    First antenna indices of shape :code:`(row,)`.
antenna2 : $(array_type)
    Second antenna indices of shape :code:`(row,)`
jones : $(array_type)
    Gain solutions of shape :code:`(time, ant, chan, dir, corr)`
    or :code:`(time, ant, chan, dir, corr, corr)`.
residual : $(array_type)
    Residual values of shape :code:`(row, chan, corr)`.
    or :code:`(row, chan, corr, corr)`.
model : $(array_type)
    Model data values of shape :code:`(row, chan, dir, corr)`
    or :code:`(row, chan, dir, corr, corr)`.
flag : $(array_type)
    Flag data of shape :code:`(row, chan, corr)`
    or :code:`(row, chan, corr, corr)`

Returns
-------
jhj : $(array_type)
    The diagonal of the Hessian of
    shape :code:`(time, ant, chan, dir, corr)`
    or :code:`(time, ant, chan, dir, corr, corr)`.
jhr : $(array_type)
    Residuals projected into signal space
    of shape :code:`(time, ant, chan, dir, corr)`
    or :code:`(time, ant, chan, dir, corr, corr)`.
""")


try:
    compute_jhj_and_jhr.__doc__ = JHJ_AND_JHR_DOCS.substitute(
                                    array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass

COMPUTE_JHJ_DOCS = DocstringTemplate("""
Computes the diagonal of the Hessian
required to perform phase-only maximum
likelihood calibration. Currently assumes
scalar or diagonal inputs.

Parameters
----------
time_bin_indices : $(array_type)
    The start indices of the time bins
    of shape :code:`(utime)`
time_bin_counts : $(array_type)
    The counts of unique time in each
    time bin of shape :code:`(utime)`
antenna1 : $(array_type)
    First antenna indices of shape :code:`(row,)`.
antenna2 : $(array_type)
    Second antenna indices of shape :code:`(row,)`
jones : $(array_type)
    Gain solutions of shape :code:`(time, ant, chan, dir, corr)`
    or :code:`(time, ant, chan, dir, corr, corr)`.
model : $(array_type)
    Model data values of shape :code:`(row, chan, dir, corr)`
    or :code:`(row, chan, dir, corr, corr)`.
flag : $(array_type)
    Flag data of shape :code:`(row, chan, corr)`
    or :code:`(row, chan, corr, corr)`

Returns
-------
jhj : $(array_type)
    The diagonal of the Hessian of
    shape :code:`(time, ant, chan, dir, corr)`
    or :code:`(time, ant, chan, dir, corr, corr)`.
""")

try:
    compute_jhj.__doc__ = COMPUTE_JHJ_DOCS.substitute(
                            array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass

COMPUTE_JHR_DOCS = DocstringTemplate("""
Computes the residual projected in to gain space.

Parameters
----------
time_bin_indices : $(array_type)
    The start indices of the time bins
    of shape :code:`(utime)`
time_bin_counts : $(array_type)
    The counts of unique time in each
    time bin of shape :code:`(utime)`
antenna1 : $(array_type)
    First antenna indices of shape :code:`(row,)`.
antenna2 : $(array_type)
    Second antenna indices of shape :code:`(row,)`
jones : $(array_type)
    Gain solutions of shape :code:`(time, ant, chan, dir, corr)`
    or :code:`(time, ant, chan, dir, corr, corr)`.
residual : $(array_type)
    Residual values of shape :code:`(row, chan, corr)`.
    or :code:`(row, chan, corr, corr)`.
model : $(array_type)
    Model data values of shape :code:`(row, chan, dir, corr)`
    or :code:`(row, chan, dir, corr, corr)`.
flag : $(array_type)
    Flag data of shape :code:`(row, chan, corr)`
    or :code:`(row, chan, corr, corr)`

Returns
-------
jhr : $(array_type)
    The residual projected into gain space
    shape :code:`(time, ant, chan, dir, corr)`
    or :code:`(time, ant, chan, dir, corr, corr)`.
""")

try:
    compute_jhr.__doc__ = COMPUTE_JHR_DOCS.substitute(
                            array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass
