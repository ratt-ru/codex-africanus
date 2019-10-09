# -*- coding: utf-8 -*-


import numpy as np
from africanus.util.docs import DocstringTemplate
from africanus.util.numba import generated_jit, njit

DIAG_DIAG = 0
DIAG = 1
FULL = 2


def check_type(jones, vis):
    """
    Determines which calibration scenario to apply i.e.
    DIAG_DIAG, DIAG or COMPLEX2x2.

    Parameters
    ----------
    jones : $(array_type)
        Jones term of shape :code:`(time, ant, chan, dir, corr)`
        or :code:`(time, ant, chan, dir, corr, corr)`
    vis : $(array_type)
        Visibility data of shape :code:`(row, chan, corr)`
        or :code:`(row, chan, corr, corr)`

    """
    vis_axes_count = vis.ndim
    jones_axes_count = jones.ndim
    if vis_axes_count == 3:
        mode = DIAG_DIAG
        if jones_axes_count != 5:
            raise RuntimeError("Jones axes not compatible with \
                                visibility axes. Expected length \
                                5 but got length %d" % jones_axes_count)

    elif vis_axes_count == 4:
        if jones_axes_count == 5:
            mode = DIAG

        elif jones_axes_count == 6:
            mode = FULL
        else:
            raise RuntimeError("Jones term has incorrect shape")
    else:
        raise RuntimeError("Visibility data has incorrect shape")

    return mode


def jones_inverse_mul_factory(mode):
    if mode == DIAG_DIAG:
        def jones_inverse_mul(a1j, blj, a2j, out):
            out[...] = blj/(a1j*np.conj(a2j))
    elif mode == DIAG:
        def jones_inverse_mul(a1j, blj, a2j, out):
            out[0, 0] = blj[0, 0]/(a1j[0]*np.conj(a2j[0]))
            out[0, 1] = blj[0, 1]/(a1j[0]*np.conj(a2j[1]))
            out[1, 0] = blj[1, 0]/(a1j[1]*np.conj(a2j[0]))
            out[1, 1] = blj[1, 1]/(a1j[1]*np.conj(a2j[1]))
    elif mode == FULL:
        def jones_inverse_mul(a1j, blj, a2j, out):
            # get determinant
            deta1j = a1j[0, 0]*a1j[1, 1]-a1j[0, 1]*a1j[1, 0]
            # compute inverse
            a00 = a1j[1, 1]/deta1j
            a01 = -a1j[1, 0]/deta1j
            a10 = -a1j[0, 1]/deta1j
            a11 = a1j[0, 0]/deta1j
            # get determinant
            deta2j = a2j[0, 0]*a2j[1, 1]-a2j[0, 1]*a2j[1, 0]
            # get conjugate transpose inverse
            b00 = np.conj(a2j[1, 1]/deta2j)
            b01 = np.conj(-a2j[1, 0]/deta2j)
            b10 = np.conj(-a2j[0, 1]/deta2j)
            b11 = np.conj(a2j[0, 0]/deta2j)
            # precompute resuable terms
            t1 = a00*blj[0, 0]
            t2 = a01*blj[1, 0]
            t3 = a00*blj[0, 1]
            t4 = a01*blj[1, 1]
            # overwrite with result
            out[0, 0] = t1*b00 +\
                t2*b00 +\
                t3*b10 +\
                t4*b10
            out[0, 1] = t1*b01 +\
                t2*b01 +\
                t3*b11 +\
                t4*b11
            t1 = a10*blj[0, 0]
            t2 = a11*blj[1, 0]
            t3 = a10*blj[0, 1]
            t4 = a11*blj[1, 1]
            out[1, 0] = t1*b00 +\
                t2*b00 +\
                t3*b10 +\
                t4*b10
            out[1, 1] = t1*b01 +\
                t2*b01 +\
                t3*b11 +\
                t4*b11
    return njit(nogil=True)(jones_inverse_mul)


def subtract_model_factory(mode):
    if mode == DIAG_DIAG:
        def subtract_model(a1j, blj, a2j, model, out):
            n_dir = np.shape(model)[0]
            out[...] = blj
            for s in range(n_dir):
                out -= a1j[s]*model[s]*np.conj(a2j[s])
    elif mode == DIAG:
        def subtract_model(a1j, blj, a2j, model, out):
            n_dir = np.shape(model)[0]
            out[...] = blj
            for s in range(n_dir):
                out[0, 0] -= a1j[s, 0]*model[s, 0, 0] * np.conj(a2j[s, 0])
                out[0, 1] -= a1j[s, 0]*model[s, 0, 1] * np.conj(a2j[s, 1])
                out[1, 0] -= a1j[s, 1]*model[s, 1, 0] * np.conj(a2j[s, 0])
                out[1, 1] -= a1j[s, 1]*model[s, 1, 1] * np.conj(a2j[s, 1])
    elif mode == FULL:
        def subtract_model(a1j, blj, a2j, model, out):
            n_dir = np.shape(model)[0]
            for s in range(n_dir):
                # precompute resuable terms
                t1 = a1j[s, 0, 0]*model[s, 0, 0]
                t2 = a1j[s, 0, 1]*model[s, 1, 0]
                t3 = a1j[s, 0, 0]*model[s, 0, 1]
                t4 = a1j[s, 0, 1]*model[s, 1, 1]
                tmp = np.conj(a2j[s].T)
                # overwrite with result
                out[0, 0] = blj[0, 0] -\
                    t1*tmp[0, 0] +\
                    t2*tmp[0, 0] +\
                    t3*tmp[1, 0] +\
                    t4*tmp[1, 0]
                out[0, 1] = blj[0, 1] -\
                    t1*tmp[0, 1] +\
                    t2*tmp[0, 1] +\
                    t3*tmp[1, 1] +\
                    t4*tmp[1, 1]
                t1 = a1j[s, 1, 0]*model[s, 0, 0]
                t2 = a1j[s, 1, 1]*model[s, 1, 0]
                t3 = a1j[s, 1, 0]*model[s, 0, 1]
                t4 = a1j[s, 1, 1]*model[s, 1, 1]
                out[1, 0] = blj[1, 0] -\
                    t1*tmp[0, 0] +\
                    t2*tmp[0, 0] +\
                    t3*tmp[1, 0] +\
                    t4*tmp[1, 0]
                out[1, 1] = blj[1, 1] -\
                    t1*tmp[0, 1] +\
                    t2*tmp[0, 1] +\
                    t3*tmp[1, 1] +\
                    t4*tmp[1, 1]
    return njit(nogil=True)(subtract_model)


@generated_jit(nopython=True, nogil=True, cache=True)
def correct_vis(time_bin_indices, time_bin_counts,
                antenna1, antenna2, jones, vis, flag):

    mode = check_type(jones, vis)

    jones_inverse_mul = jones_inverse_mul_factory(mode)

    def _correct_vis_fn(time_bin_indices, time_bin_counts,
                        antenna1, antenna2, jones, vis, flag):
        jones_shape = np.shape(jones)
        n_tim = jones_shape[0]
        n_dir = jones_shape[3]
        if n_dir > 1:
            raise ValueError("Jones has n_dir > 1.\
                                Cannot correct for direction dependent gains")
        n_chan = jones_shape[2]
        corrected_vis = np.zeros_like(vis, dtype=vis.dtype)
        for t in range(n_tim):
            for row in range(time_bin_indices[t],
                             time_bin_indices[t] + time_bin_counts[t]):
                p = int(antenna1[row])
                q = int(antenna2[row])
                gp = jones[t, p]
                gq = jones[t, q]
                for nu in range(n_chan):
                    if not np.any(flag[row, nu]):
                        jones_inverse_mul(
                            gp[nu, 0], vis[row, nu], gq[nu, 0],
                            corrected_vis[row, nu])
        return corrected_vis

    return _correct_vis_fn


@generated_jit(nopython=True, nogil=True, cache=True)
def residual_vis(time_bin_indices, time_bin_counts, antenna1,
                 antenna2, jones, vis, flag, model):

    mode = check_type(jones, vis)
    subtract_model = subtract_model_factory(mode)

    def _residual_vis_fn(time_bin_indices, time_bin_counts, antenna1,
                         antenna2, jones, vis, flag, model):
        n_tim = np.shape(time_bin_indices)[0]
        vis_shape = np.shape(vis)
        n_chan = vis_shape[1]
        residual = np.zeros(vis_shape, dtype=vis.dtype)
        for t in range(n_tim):
            for row in range(time_bin_indices[t],
                             time_bin_indices[t] + time_bin_counts[t]):
                p = int(antenna1[row])
                q = int(antenna2[row])
                gp = jones[t, p]
                gq = jones[t, q]
                for nu in range(n_chan):
                    if not np.any(flag[row, nu]):
                        subtract_model(
                            gp[nu], vis[row, nu], gq[nu],
                            model[row, nu], residual[row, nu])
        return residual

    return _residual_vis_fn


RESIDUAL_VIS_DOCS = DocstringTemplate("""
Computes residual visibilities in place
given model visibilities and gains solutions.

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
vis : $(array_type)
    Data values of shape :code:`(row, chan, corr)`.
    or :code:`(row, chan, corr, corr)`.
flag : $(array_type)
    Flag data of shape :code:`(row, chan, corr)`
    or :code:`(row, chan, corr, corr)`
model : $(array_type)
    Model data values of shape :code:`(row, chan, dir, corr)`
    or :code:`(row, chan, dir, corr, corr)`.

Returns
-------
residual : $(array_type)
    Residual visibilities of shape
    :code:`(time, ant, chan, dir, corr)`
    or :code:`(time, ant, chan, dir, corr, corr)`.
""")


try:
    residual_vis.__doc__ = RESIDUAL_VIS_DOCS.substitute(
                                    array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass

CORRECT_VIS_DOCS = DocstringTemplate("""
Apply DIE gains to visibilities to
generate corrected visibilities. For a
measurement model of the form

.. math::


    V_{pq} = G_{p} X_{pq} G_{q}^H + n_{pq}

the corrected visibilities are defined as

.. math::

    C_{pq} = G_{p}^{-1} V_{pq} G_{q}^{-H}

The corrected visibilities therefore have
a non-trivial noise contribution. Note
it is only possible to form corrected
data from direction independent gains
solutions so the :code:`dir` axis on
the jones terms should always be one.

Parameters
----------
time_bin_indices : $(array_type)
    The start indices of the time bins
    of shape :code:`(utime)`.
time_bin_counts : $(array_type)
    The counts of unique time in each
    time bin of shape :code:`(utime)`.
antenna1 : $(array_type)
    Antenna 1 index used to look up the antenna Jones
    for a particular baseline with shape :code:`(row,)`.
antenna2 : $(array_type)
    Antenna 2 index used to look up the antenna Jones
    for a particular baseline with shape :code:`(row,)`.
jones : $(array_type)
    Gain solutions of shape :code:`(time, ant, chan, dir, corr)`
    or :code:`(time, ant, chan, dir, corr, corr)`.
vis : $(array_type)
    Data values of shape :code:`(row, chan, corr)`
    or :code:`(row, chan, corr, corr)`.
flag : $(array_type)
    Flag data of shape :code:`(row, chan, corr)`
    or :code:`(row, chan, corr, corr)`.

Returns
-------
corrected_vis : $(array_type)
    True visibilities of shape :code:`(row,chan,corr_1,corr_2)`
""")

try:
    correct_vis.__doc__ = CORRECT_VIS_DOCS.substitute(
                        array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass
