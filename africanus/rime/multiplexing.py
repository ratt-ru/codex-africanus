# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numba
import numpy as np

from ..util.requirements import have_packages, MissingPackageException
from ..util.docs import on_rtd


@numba.njit(nogil=True)
def _check_shapes(ant1, bl, ant2, out):
    if ant1.shape != ant2.shape:
        raise ValueError("ant1.shape != ant2.shape")

    if ant1.shape != bl.shape:
        raise ValueError("ant1.shape != bl.shape")

    if bl.shape != out.shape:
        raise ValueError("bl.shape != out.shape")


@numba.njit(nogil=True)
def jones_2x2_mul(ant1, bl, ant2, out):
    _check_shapes(ant1, bl, ant2, out)

    reshape = (-1,) + ant1.shape[-2:]
    fa1 = np.reshape(ant1, reshape)
    fa2 = np.reshape(ant2, reshape)
    fb = np.reshape(bl, reshape)
    fr = np.reshape(out, reshape)

    for r in range(fa1.shape[0]):
        a2_xx_conj = np.conj(fa2[r, 0, 0])
        a2_xy_conj = np.conj(fa2[r, 0, 1])
        a2_yx_conj = np.conj(fa2[r, 1, 0])
        a2_yy_conj = np.conj(fa2[r, 1, 1])

        xx = fb[r, 0, 0] * a2_xx_conj + fb[r, 0, 1] * a2_yx_conj
        xy = fb[r, 0, 0] * a2_xy_conj + fb[r, 0, 1] * a2_yy_conj
        yx = fb[r, 1, 0] * a2_xx_conj + fb[r, 1, 1] * a2_yx_conj
        yy = fb[r, 1, 0] * a2_xy_conj + fb[r, 1, 1] * a2_yy_conj

        fr[r, 0, 0] = fa1[r, 0, 0] * xx + fa1[r, 0, 1] * yx
        fr[r, 0, 1] = fa1[r, 0, 0] * xy + fa1[r, 0, 1] * yy
        fr[r, 1, 0] = fa1[r, 1, 0] * xx + fa1[r, 1, 1] * yx
        fr[r, 1, 1] = fa1[r, 1, 0] * xy + fa1[r, 1, 1] * yy


@numba.njit(nogil=True)
def jones_1_and_2_mul(ant1, bl, ant2, out):
    _check_shapes(ant1, bl, ant2, out)

    reshape = (-1,) + ant1.shape[-2:]
    fa1 = np.reshape(ant1, reshape)
    fa2 = np.reshape(ant2, reshape)
    fb = np.reshape(bl, reshape)
    fr = np.reshape(out, reshape)

    for r in range(fa1.shape[0]):
        for c in range(fa2.shape[1]):
            fr[r, c] = fa1[r, c] * fb[r, c] * np.conj(fa2[r, c])


@numba.njit(nogil=True)
def _multiplex(time_index, ant1, ant2,
               ant1_jones, ant2_jones, row_jones,
               g1_jones, g2_jones, out, jones_mul):

    jones_out = np.zeros(out.shape[2:], out.dtype)

    # Iterate over sources
    for s in range(ant1_jones.shape[0]):
        # Iterate over rows
        for r, (ti, a1, a2) in enumerate(zip(time_index, ant1, ant2)):
            # Iterate over channels
            for c in range(ant1_jones.shape[3]):
                jones_mul(ant1_jones[s, ti, a1, c], row_jones[s, r, c],
                          ant2_jones[s, ti, a2, c], jones_out)

                # Accumulate over sources
                out[r, c] += jones_out

    # Iterate over rows
    for r, (ti, a1, a2) in enumerate(zip(time_index, ant1, ant2)):
        # Iterate over channels
        for c in range(ant1_jones.shape[3]):
            jones_mul(g1_jones[ti, a1, c], out[r, c],
                      g2_jones[ti, a2, c], jones_out)

            out[r, c] = jones_out


def multiplex(time_index, antenna1, antenna2,
              ant1_jones, ant2_jones, row_jones,
              g1_jones, g2_jones):
    """
    Multiply jones terms together to form model visibilities according
    to the following formula:

    .. math::

        V_{pq} = G_{p} \\left( \\sum_{s} A_{ps} B_{pqs} A_{qs}^H \\right) G_{q}^H

    where for antenna :math:`p` and :math:`q`, and source :math:`s`:

    - :math:`A_{ps}` represents per-antenna and source Jones terms.
    - :math:`B_{pqs}` represents the per-baseline and source Jones terms.
    - :math:`G_{p}` represents per-antenna Jones terms.

    Generally, :math:`A_{ps}` and :math:`G_{p}` should be formed by creating
    Jones terms using RIME functions in the `RIME API <rime-api-anchor_>`_
    and combining them together with a function like :func:`numpy.einsum`

    Parameters
    ----------
    time_index : :class:`numpy.ndarray`
        shape :code:`(row,)`
    antenna1 : :class:`numpy.ndarray`
        shape :code:`(row,)`
    antenna2 : :class:`numpy.ndarray`
        shape :code:`(row,)`
    ant1_jones : :class:`numpy.ndarray`
        shape :code:`(source,time,ant,chan,corr_1,corr_2)`
    ant2_jones : :class:`numpy.ndarray`
        shape :code:`(source,time,ant,chan,corr_1,corr_2)`
    row_jones : class:`numpy.ndarray`
        shape :code:`(source,row,chan,corr_1,corr_2)`
    g1_jones : :class:`numpy.ndarray`
        shape :code:`(time,ant,chan,corr_1,corr_2)`
    g2_jones : :class:`numpy.ndarray`
        shape :code:`(time,ant,chan,corr_1,corr_2)`

    Returns
    -------
    :class:`numpy.ndarray`
        Model visibilities of shape :code:`(row,chan,corr_1,corr_2)`
    """

    _, row, chan = row_jones.shape[:3]
    corrs = row_jones.shape[3:]

    if corrs == (1,) or corrs == (2,):
        jones_mul = jones_1_and_2_mul
    elif corrs == (2, 2):
        jones_mul = jones_2x2_mul
    else:
        raise ValueError("Unhandled correlations %s", (corrs,))

    out = np.zeros((row, chan) + corrs, dtype=row_jones.dtype)

    _multiplex(time_index, antenna1, antenna2,
               ant1_jones, ant2_jones, row_jones,
               g1_jones, g2_jones, out, jones_mul)

    return out
