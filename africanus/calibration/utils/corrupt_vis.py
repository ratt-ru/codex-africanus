# -*- coding: utf-8 -*-

import numpy as np
from africanus.util.docs import DocstringTemplate
from africanus.util.numba import generated_jit, njit
from africanus.calibration.utils import check_type
from africanus.calibration.utils.utils import DIAG_DIAG, DIAG, FULL


def jones_mul_factory(mode):
    if mode == DIAG_DIAG:
        def jones_mul(a1j, model, a2j, out):
            n_dir = np.shape(model)[0]
            for s in range(n_dir):
                out += a1j[s]*model[s]*np.conj(a2j[s])
    elif mode == DIAG:
        def jones_mul(a1j, model, a2j, out):
            n_dir = np.shape(model)[0]
            for s in range(n_dir):
                out[0, 0] += a1j[s, 0]*model[s, 0, 0] * np.conj(a2j[s, 0])
                out[0, 1] += a1j[s, 0]*model[s, 0, 1] * np.conj(a2j[s, 1])
                out[1, 0] += a1j[s, 1]*model[s, 1, 0] * np.conj(a2j[s, 0])
                out[1, 1] += a1j[s, 1]*model[s, 1, 1] * np.conj(a2j[s, 1])
    elif mode == FULL:
        def jones_mul(a1j, model, a2j, out):
            n_dir = np.shape(model)[0]
            for s in range(n_dir):
                # precompute resuable terms
                t1 = a1j[s, 0, 0]*model[s, 0, 0]
                t2 = a1j[s, 0, 1]*model[s, 1, 0]
                t3 = a1j[s, 0, 0]*model[s, 0, 1]
                t4 = a1j[s, 0, 1]*model[s, 1, 1]
                tmp = np.conj(a2j[s].T)
                # overwrite with result
                out[0, 0] += t1*tmp[0, 0] +\
                    t2*tmp[0, 0] +\
                    t3*tmp[1, 0] +\
                    t4*tmp[1, 0]
                out[0, 1] += t1*tmp[0, 1] +\
                    t2*tmp[0, 1] +\
                    t3*tmp[1, 1] +\
                    t4*tmp[1, 1]
                t1 = a1j[s, 1, 0]*model[s, 0, 0]
                t2 = a1j[s, 1, 1]*model[s, 1, 0]
                t3 = a1j[s, 1, 0]*model[s, 0, 1]
                t4 = a1j[s, 1, 1]*model[s, 1, 1]
                out[1, 0] += t1*tmp[0, 0] +\
                    t2*tmp[0, 0] +\
                    t3*tmp[1, 0] +\
                    t4*tmp[1, 0]
                out[1, 1] += t1*tmp[0, 1] +\
                    t2*tmp[0, 1] +\
                    t3*tmp[1, 1] +\
                    t4*tmp[1, 1]

    return njit(nogil=True, inline='always')(jones_mul)


@generated_jit(nopython=True, nogil=True, cache=True)
def corrupt_vis(time_bin_indices, time_bin_counts, antenna1,
                antenna2, jones, model):

    mode = check_type(jones, model, vis_type='model')
    jones_mul = jones_mul_factory(mode)

    def _corrupt_vis_fn(time_bin_indices, time_bin_counts, antenna1,
                        antenna2, jones, model):
        # for dask arrays we need to adjust the chunks to
        # start counting from zero
        time_bin_indices -= time_bin_indices.min()
        n_tim = np.shape(time_bin_indices)[0]
        model_shape = np.shape(model)
        vis_shape = model.shape[:2] + model.shape[3:]
        vis = np.zeros(vis_shape, dtype=model.dtype)
        n_chan = model_shape[1]
        for t in range(n_tim):
            for row in range(time_bin_indices[t],
                             time_bin_indices[t] + time_bin_counts[t]):
                p = int(antenna1[row])
                q = int(antenna2[row])
                gp = jones[t, p]
                gq = jones[t, q]
                for nu in range(n_chan):
                    jones_mul(gp[nu], model[row, nu], gq[nu], vis[row, nu])
        return vis

    return _corrupt_vis_fn


CORRUPT_VIS_DOCS = DocstringTemplate("""
Corrupts model visibilities with arbitrary
Jones terms.

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
    Gains of shape :code:`(time, ant, chan, dir, corr)`
    or :code:`(time, ant, chan, dir, corr, corr)`.
model : $(array_type)
    Model data values of shape :code:`(row, chan, dir, corr)`
    or :code:`(row, chan, dir, corr, corr)`.

Returns
-------
vis : $(array_type)
    visibilities of shape
    :code:`(time, ant, chan, dir, corr)`
    or :code:`(time, ant, chan, dir, corr, corr)`.
""")


try:
    corrupt_vis.__doc__ = CORRUPT_VIS_DOCS.substitute(
                                    array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass
