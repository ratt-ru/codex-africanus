# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import numba
from numba import types, generated_jit, njit
import numpy as np

from ..util.docs import doc_tuple_to_str


def jones_mul_factory(have_ants, have_bl, jones_type):
    ex = ValueError("Invalid Jones Type %s" % jones_type)

    if have_bl and have_ants:
        if jones_type == JONES_1_OR_2:
            def jones_mul(a1j, blj, a2j, out):
                for c in range(out.shape[0]):
                    out[c] = a1j[c] * blj[c] * np.conj(a2j[c])

        elif jones_type == JONES_2X2:
            def jones_mul(a1j, blj, a2j, out):
                a2_xx_conj = np.conj(a2j[0, 0])
                a2_xy_conj = np.conj(a2j[0, 1])
                a2_yx_conj = np.conj(a2j[1, 0])
                a2_yy_conj = np.conj(a2j[1, 1])

                xx = blj[0, 0] * a2_xx_conj + blj[0, 1] * a2_yx_conj
                xy = blj[0, 0] * a2_xy_conj + blj[0, 1] * a2_yy_conj
                yx = blj[1, 0] * a2_xx_conj + blj[1, 1] * a2_yx_conj
                yy = blj[1, 0] * a2_xy_conj + blj[1, 1] * a2_yy_conj

                out[0, 0] = a1j[0, 0] * xx + a1j[0, 1] * yx
                out[0, 1] = a1j[0, 0] * xy + a1j[0, 1] * yy
                out[1, 0] = a1j[1, 0] * xx + a1j[1, 1] * yx
                out[1, 1] = a1j[1, 0] * xy + a1j[1, 1] * yy

        else:
            raise ex
    elif have_ants and not have_bl:
        if jones_type == JONES_1_OR_2:
            def jones_mul(a1j, a2j, out):
                for c in range(out.shape[0]):
                    out[c] = a1j[c] * np.conj(a2j[c])

        elif jones_type == JONES_2X2:
            def jones_mul(a1j, a2j, out):
                a2_xx_conj = np.conj(a2j[0, 0])
                a2_xy_conj = np.conj(a2j[0, 1])
                a2_yx_conj = np.conj(a2j[1, 0])
                a2_yy_conj = np.conj(a2j[1, 1])

                out[0, 0] = a1j[0, 0] * a2_xx_conj + a1j[0, 1] * a2_yx_conj
                out[0, 1] = a1j[0, 0] * a2_xy_conj + a1j[0, 1] * a2_yy_conj
                out[1, 0] = a1j[1, 0] * a2_xx_conj + a1j[1, 1] * a2_yx_conj
                out[1, 1] = a1j[1, 0] * a2_xy_conj + a1j[1, 1] * a2_yy_conj
        else:
            raise ex
    elif not have_ants and have_bl:
        if jones_type == JONES_1_OR_2:
            def jones_mul(blj, out):
                for c in range(out.shape[0]):
                    out[c] = blj[c]
        elif jones_type == JONES_2X2:
            def jones_mul(blj, out):
                out[0, 0] = blj[0, 0]
                out[0, 1] = blj[0, 1]
                out[1, 0] = blj[1, 0]
                out[1, 1] = blj[1, 1]
        else:
            raise ex
    else:
        raise ValueError("Invalid Permutation: "
                         "have_ants %s "
                         "have_bl %s" % (have_ants, have_bl))

    return njit(nogil=True, cache=True)(jones_mul)


def sum_coherencies_factory(have_ants, have_bl, jones_type):
    jones_mul = jones_mul_factory(have_ants, have_bl, jones_type)

    if have_ants and have_bl:
        def sum_coh_fn(time, ant1, ant2, a1j, blj, a2j, out):
            jones_out = np.empty(out.shape[2:], dtype=out.dtype)

            for s in range(a1j.shape[0]):
                for r, (ti, a1, a2) in enumerate(zip(time, ant1, ant2)):
                    for f in range(a1j.shape[3]):
                        jones_mul(a1j[s, ti, a1, f],
                                  blj[s, r, f],
                                  a2j[s, ti, a2, f],
                                  jones_out)

                        out[r, f] += jones_out

    elif have_ants and not have_bl:
        def sum_coh_fn(time, ant1, ant2, a1j, blj, a2j, out):
            jones_out = np.empty(out.shape[2:], dtype=out.dtype)

            for s in range(a1j.shape[0]):
                for r, (ti, a1, a2) in enumerate(zip(time, ant1, ant2)):
                    for f in range(a1j.shape[3]):
                        jones_mul(a1j[s, ti, a1, f],
                                  a2j[s, ti, a2, f],
                                  jones_out)

                        out[r, f] += jones_out
    elif not have_ants and have_bl:
        def sum_coh_fn(time, ant1, ant2, a1j, blj, a2j, out):
            for s in range(blj.shape[0]):
                for r in range(blj.shape[1]):
                    for f in range(blj.shape[2]):
                        out[r, f] += blj[s, r, f]
    else:
        raise ValueError("Invalid Combination")

    return njit(nogil=True, cache=True)(sum_coh_fn)


def output_factory(have_ants, have_bl):
    if have_ants:
        def output(time_index, ant1_jones, bl_jones, ant2_jones):
            row = time_index.shape[0]
            chan = ant1_jones.shape[3]
            corrs = ant1_jones.shape[4:]
            return np.zeros((row, chan) + corrs, dtype=ant1_jones.dtype)
    elif have_bl:
        def output(time_index, ant1_jones, bl_jones, ant2_jones):
            row = time_index.shape[0]
            chan = bl_jones.shape[2]
            corrs = bl_jones.shape[3:]
            return np.zeros((row, chan) + corrs, dtype=bl_jones.dtype)
    else:
        raise ValueError("Insufficient inputs were supplied "
                         "for determining the output shape")

    return njit(nogil=True, cache=True)(output)


def apply_dies_factory(have_dies, jones_type):
    if have_dies:
        jones_mul = jones_mul_factory(True, True, jones_type)

        def apply_dies(time, ant1, ant2, g1_jones, g2_jones, out):
            jones_out = np.empty(out.shape[2:], dtype=out.dtype)

            # Iterate over rows
            for r, (ti, a1, a2) in enumerate(zip(time, ant1, ant2)):
                # Iterate over channels
                for c in range(out.shape[1]):
                    jones_mul(g1_jones[ti, a1, c], out[r, c],
                              g2_jones[ti, a2, c], jones_out)

                    # Write output of multiplication back
                    out[r, c] = jones_out

    else:
        def apply_dies(time, ant1, ant2, g1_jones, g2_jones, out):
            pass

    return njit(nogil=True, cache=True)(apply_dies)


JONES_NOT_PRESENT = 0
JONES_1_OR_2 = 1
JONES_2X2 = 2


def _get_jones_types(present, name, ndarray_type, corr_1_dims, corr_2_dims):
    if not present:
        return JONES_NOT_PRESENT
    if ndarray_type.ndim == corr_1_dims:
        return JONES_1_OR_2
    elif ndarray_type.ndim == corr_2_dims:
        return JONES_2X2
    else:
        raise ValueError("%s.ndim not in (%d, %d)" %
                         (name, corr_1_dims, corr_2_dims))



@generated_jit(nopython=True, nogil=True, cache=True)
def predict_vis(time_index, antenna1, antenna2,
                ant1_jones, row_jones, ant2_jones,
                g1_jones, g2_jones):

    have_a1 = not isinstance(ant1_jones, types.misc.NoneType)
    have_a2 = not isinstance(ant2_jones, types.misc.NoneType)
    have_bl = not isinstance(row_jones, types.misc.NoneType)
    have_g1 = not isinstance(g1_jones, types.misc.NoneType)
    have_g2 = not isinstance(g2_jones, types.misc.NoneType)

    assert time_index.ndim == 1
    assert antenna1.ndim == 1
    assert antenna2.ndim == 1

    if have_a1 ^ have_a2:
        raise ValueError("Both ant1_jones and ant2_jones "
                         "must be present or absent")

    if have_g1 ^ have_g2:
        raise ValueError("Both g1_jones and g2_jones "
                         "must be present or absent")


    have_ants = have_a1 and have_a2
    have_dies = have_g1 and have_g2

    if not (have_ants or have_bl):
        raise ValueError("No Jones Terms were supplied")

    if have_a1:
        assert ant1_jones.ndim in (5, 6)

    if have_a2:
        assert ant2_jones.ndim in (5, 6)

    if have_bl:
        assert row_jones.ndim in (4, 5)

    if have_g1:
        assert g1_jones.ndim in (4, 5)

    if have_g1:
        assert g2_jones.ndim in (4, 5)

    jones_types = [
        _get_jones_types(have_a1, "ant1_jones", ant1_jones, 5, 6),
        _get_jones_types(have_bl, "row_jones", row_jones, 4, 5),
        _get_jones_types(have_a2, "ant2_jones", ant2_jones, 5, 6)]

    ptypes = [t for t in jones_types if t != JONES_NOT_PRESENT]

    if not all(ptypes[0] == p for p in ptypes[1:]):
        raise ValueError("Jones Matrix Correlations were mismatched")

    try:
        jones_type = ptypes[0]
    except IndexError:
        raise ValueError("Illegal Condition: No Jones Matrices were supplied")


    out_fn = output_factory(have_ants, have_bl)
    sum_coh_fn = sum_coherencies_factory(have_ants, have_bl, jones_type)
    apply_dies_fn = apply_dies_factory(have_dies, jones_type)

    def fn(time_index, antenna1, antenna2,
           ant1_jones, row_jones, ant2_jones,
           g1_jones, g2_jones):

        out = out_fn(time_index, ant1_jones, row_jones, ant2_jones)
        sum_coh_fn(time_index, antenna1, antenna2,
                   ant1_jones, row_jones, ant2_jones, out)
        apply_dies_fn(time_index, antenna1, antenna2,
                      g1_jones, g2_jones, out)

        return out

    return fn
