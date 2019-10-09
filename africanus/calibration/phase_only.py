# -*- coding: utf-8 -*-


import numpy as np
from africanus.calibration.utils import check_type
from africanus.util.docs import DocstringTemplate
from africanus.calibration.utils import residual_vis
from africanus.util.numba import generated_jit, njit

DIAG_DIAG = 0
DIAG = 1
FULL = 2


def jacobian_factory(mode):
    if mode == DIAG_DIAG:
        def jacobian(a1j, blj, a2j, sign, out):
            out[...] = sign * a1j * blj * np.conj(a2j)
    elif mode == DIAG:
        def jacobian(a1j, blj, a2j, sign, out):
            out[...] = 0
    elif mode == FULL:
        def jacobian(a1j, blj, a2j, sign, out):
            out[...] = 0
    return njit(nogil=True)(jacobian)


@generated_jit(nopython=True, nogil=True, cache=True)
def jhj_and_jhr(time_bin_indices, time_bin_counts, antenna1,
                antenna2, jones, residual, model, flag):

    mode = check_type(jones, residual)

    if mode:
        raise NotImplementedError("Only DIAG-DIAG case has been implemented")

    jacobian = jacobian_factory(mode)

    def _jhj_and_jhr_fn(time_bin_indices, time_bin_counts, antenna1,
                        antenna2, jones, residual, model, flag):
        jones_shape = np.shape(jones)
        tmp_out_array = np.zeros_like(jones[0, 0, 0, 0], dtype=jones.dtype)
        n_tim = jones_shape[0]
        n_ant = jones_shape[1]
        n_chan = jones_shape[2]
        n_dir = jones_shape[3]

        jhr = np.zeros(jones.shape, dtype=jones.dtype)
        jhj = np.zeros(jones.shape, dtype=jones.real.dtype)

        for t in range(n_tim):
            ind = np.arange(time_bin_indices[t],
                            time_bin_indices[t] + time_bin_counts[t])
            for ant in range(n_ant):
                # find where either antenna == ant
                # these will be mutually exclusive since no autocorrs
                for row in ind:
                    if antenna1[row] == ant or antenna2[row] == ant:
                        p = antenna1[row]
                        q = antenna2[row]
                        if ant == p:
                            sign = 1.0j
                        elif ant == q:
                            sign = -1.0j
                        else:
                            raise ValueError(
                                "Got impossible antenna number. This is a bug")
                        for nu in range(n_chan):
                            if not np.any(flag[row, nu]):
                                for s in range(n_dir):
                                    jacobian(
                                        jones[t, p, nu, s],
                                        model[row, nu, s],
                                        jones[t, q, nu, s],
                                        sign,
                                        tmp_out_array)
                                    jhj[t, ant, nu,
                                        s] += (np.conj(tmp_out_array) *
                                               tmp_out_array).real
                                    jhr[t, ant, nu,
                                        s] += (np.conj(tmp_out_array) *
                                               residual[row, nu])
        return jhj, jhr
    return _jhj_and_jhr_fn


def phase_only_gauss_newton(time_bin_indices, time_bin_counts, antenna1,
                            antenna2, jones, vis, flag, model,
                            weight, tol=1e-4, maxiter=100):
    # whiten data
    sqrtweights = np.sqrt(weight)
    vis *= sqrtweights
    model *= sqrtweights[:, :, None]

    eps = 1.0
    k = 0
    while eps > tol and k < maxiter:
        # keep track of old phases
        phases = np.angle(jones)

        # get residual
        residual = residual_vis(time_bin_indices, time_bin_counts, antenna1,
                                antenna2, jones, vis, flag, model)

        # get diag(jhj) and jhr
        jhj, jhr = jhj_and_jhr(time_bin_indices, time_bin_counts, antenna1,
                               antenna2, jones, residual, model, flag)

        # implement update
        phases_new = phases + (jhr/jhj).real
        jones = np.exp(1.0j * phases_new)

        # check convergence/iteration control
        eps = np.abs(phases_new - phases).max()
        k += 1

    return jones, jhj, jhr, k


PHASE_CALIBRATION_DOCS = DocstringTemplate("""
Performs phase-only maximum likelihood
calibration assuming scalar or diagonal
inputs using Gauss-Newton oprimisation.

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
tol : float, optional
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
k : int
    Number of iterations (will equal maxiter if
    not converged)
""")


try:
    phase_only_gauss_newton.__doc__ = PHASE_CALIBRATION_DOCS.substitute(
                                    array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass

JHJ_AND_JHR_DOCS = DocstringTemplate("""
Computes the diagonal of the Hessian and
the residual projected in to gain space.
These are the terms required to perform
phase-only maximum likelihood calibration
assuming scalar or diagonal inputs.

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
    jhj_and_jhr.__doc__ = JHJ_AND_JHR_DOCS.substitute(
                                    array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass
