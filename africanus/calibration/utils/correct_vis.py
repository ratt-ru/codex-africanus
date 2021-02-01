# -*- coding: utf-8 -*-

import numpy as np
from africanus.util.docs import DocstringTemplate
from africanus.util.numba import generated_jit, njit
from africanus.calibration.utils import check_type
from africanus.calibration.utils.utils import DIAG_DIAG, DIAG, FULL


def jones_inverse_mul_factory(mode):
    if mode == DIAG_DIAG:
        def jones_inverse_mul(a1j, blj, a2j, out):
            for c in range(out.shape[-1]):
                out[c] = blj[c]/(a1j[c]*np.conj(a2j[c]))
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
            a01 = -a1j[0, 1]/deta1j
            a10 = -a1j[1, 0]/deta1j
            a11 = a1j[0, 0]/deta1j

            # get determinant
            a2j = np.conj(a2j)
            deta2j = a2j[0, 0]*a2j[1, 1]-a2j[0, 1]*a2j[1, 0]
            # get conjugate transpose inverse
            b00 = a2j[1, 1]/deta2j
            b01 = -a2j[1, 0]/deta2j
            b10 = -a2j[0, 1]/deta2j
            b11 = a2j[0, 0]/deta2j

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
    return njit(nogil=True, inline='always')(jones_inverse_mul)


@generated_jit(nopython=True, nogil=True, cache=True)
def correct_vis(time_bin_indices, time_bin_counts,
                antenna1, antenna2, jones, vis, flag):

    mode = check_type(jones, vis)
    jones_inverse_mul = jones_inverse_mul_factory(mode)

    def _correct_vis_fn(time_bin_indices, time_bin_counts,
                        antenna1, antenna2, jones, vis, flag):
        # for dask arrays we need to adjust the chunks to
        # start counting from zero
        time_bin_indices -= time_bin_indices.min()
        jones_shape = np.shape(jones)
        n_tim = jones_shape[0]
        n_dir = jones_shape[3]
        if n_dir > 1:
            raise ValueError("Jones has n_dir > 1. Cannot correct "
                             "for direction dependent gains")
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
                        jones_inverse_mul(gp[nu, 0], vis[row, nu], gq[nu, 0],
                                          corrected_vis[row, nu])
        return corrected_vis

    return _correct_vis_fn


CORRECT_VIS_DOCS = DocstringTemplate("""
Apply inverse of direction independent gains to
visibilities to generate corrected visibilities.
For a measurement model of the form

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
