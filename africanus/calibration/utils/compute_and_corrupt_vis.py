# -*- coding: utf-8 -*-

import numpy as np
from africanus.util.docs import DocstringTemplate
from africanus.util.numba import generated_jit, njit
from africanus.calibration.utils import check_type
from africanus.constants import minus_two_pi_over_c as m2pioc
from africanus.calibration.utils.utils import DIAG_DIAG, DIAG, FULL


def jones_mul_factory(mode):
    if mode == DIAG_DIAG:
        def jones_mul(a1j, model, a2j, uvw, freq, lm, out):
            n_dir = np.shape(model)[0]
            u, v, w = uvw
            for s in range(n_dir):
                l, m = lm[s]
                n = np.sqrt(1 - l**2 - m**2)
                real_phase = m2pioc * freq * (u*l + v*m + w*(n-1))
                source_vis = model[s] * np.exp(1.0j*real_phase)/n
                for c in range(out.shape[-1]):
                    out[c] += a1j[s, c]*source_vis[c]*np.conj(a2j[s, c])
    elif mode == DIAG:
        def jones_mul(a1j, model, a2j, uvw, freq, lm, out):
            n_dir = np.shape(model)[0]
            u, v, w = uvw
            for s in range(n_dir):
                l, m = lm[s]
                n = np.sqrt(1 - l**2 - m**2)
                real_phase = m2pioc * freq * (u*l + v*m + w*(n-1))
                source_vis = model[s] * np.exp(1.0j*real_phase)/n
                out[0, 0] += a1j[s, 0]*source_vis[0, 0] * np.conj(a2j[s, 0])
                out[0, 1] += a1j[s, 0]*source_vis[0, 1] * np.conj(a2j[s, 1])
                out[1, 0] += a1j[s, 1]*source_vis[1, 0] * np.conj(a2j[s, 0])
                out[1, 1] += a1j[s, 1]*source_vis[1, 1] * np.conj(a2j[s, 1])
    elif mode == FULL:
        def jones_mul(a1j, model, a2j, uvw, freq, lm, out):
            n_dir = np.shape(model)[0]
            u, v, w = uvw
            for s in range(n_dir):
                l, m = lm[s]
                n = np.sqrt(1 - l**2 - m**2)
                real_phase = m2pioc * freq * (u*l + v*m + w*(n-1))
                source_vis = model[s] * np.exp(1.0j*real_phase)/n
                # precompute resuable terms
                t1 = a1j[s, 0, 0]*source_vis[0, 0]
                t2 = a1j[s, 0, 1]*source_vis[1, 0]
                t3 = a1j[s, 0, 0]*source_vis[0, 1]
                t4 = a1j[s, 0, 1]*source_vis[1, 1]
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
                t1 = a1j[s, 1, 0]*source_vis[0, 0]
                t2 = a1j[s, 1, 1]*source_vis[1, 0]
                t3 = a1j[s, 1, 0]*source_vis[0, 1]
                t4 = a1j[s, 1, 1]*source_vis[1, 1]
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
def compute_and_corrupt_vis(time_bin_indices, time_bin_counts, antenna1,
                            antenna2, jones, model, uvw, freq, lm):

    mode = check_type(jones, model, vis_type='model')
    jones_mul = jones_mul_factory(mode)

    def _compute_and_corrupt_vis_fn(time_bin_indices, time_bin_counts,
                                    antenna1, antenna2, jones, model,
                                    uvw, freq, lm):
        # for dask arrays we need to adjust the chunks to
        # start counting from zero
        time_bin_indices -= time_bin_indices.min()
        n_tim = np.shape(time_bin_indices)[0]
        model_shape = np.shape(model)
        vis_shape = (antenna1.shape[0],) + (freq.shape[0],) + model.shape[3:]
        vis = np.zeros(vis_shape, dtype=jones.dtype)
        n_chan = model_shape[1]
        for t in range(n_tim):
            for row in range(time_bin_indices[t],
                             time_bin_indices[t] + time_bin_counts[t]):
                p = int(antenna1[row])
                q = int(antenna2[row])
                gp = jones[t, p]
                gq = jones[t, q]
                for nu in range(n_chan):
                    jones_mul(gp[nu], model[t, nu], gq[nu], uvw[row],
                              freq[nu], lm[t], vis[row, nu])
        return vis

    return _compute_and_corrupt_vis_fn


COMPUTE_AND_CORRUPT_VIS_DOCS = DocstringTemplate("""
Corrupts time variable component model with arbitrary
Jones terms. Currrently only time variable point source
models are supported.

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
    Gains of shape :code:`(utime, ant, chan, dir, corr)`
    or :code:`(utime, ant, chan, dir, corr, corr)`.
model : $(array_type)
    Model image as a function of time with shape
    :code:`(utime, chan, dir, corr)` or
    :code:`(utime, chan, dir, corr, corr)`.
uvw : $(array_type)
    uvw coordinates of shape :code:`(row, 3)`
lm : $(array_type)
    Source lm coordinates as a function of time
    :code:`(utime, dir, 2)`

Returns
-------
vis : $(array_type)
    visibilities of shape
    :code:`(row, chan, corr)`
    or :code:`(row, chan, corr, corr)`.
""")


try:
    compute_and_corrupt_vis.__doc__ = COMPUTE_AND_CORRUPT_VIS_DOCS.substitute(
                                    array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass
