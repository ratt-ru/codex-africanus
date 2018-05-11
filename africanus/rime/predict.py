# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import numba
import numpy as np

from ..util.requirements import have_packages, MissingPackageException
from ..util.docs import on_rtd, doc_tuple_to_str


@numba.njit(nogil=True, cache=True)
def _check_shapes(ant1, bl, ant2, out):
    if ant1.shape != ant2.shape:
        raise ValueError("ant1.shape != ant2.shape")

    if ant1.shape != bl.shape:
        raise ValueError("ant1.shape != bl.shape")

    if bl.shape != out.shape:
        raise ValueError("bl.shape != out.shape")


@numba.njit(nogil=True, cache=True)
def jones_1_and_2_mul(ant1, bl, ant2, out):
    for c in range(ant1.shape[0]):
        out[c] = ant1[c] * bl[c] * np.conj(ant2[c])


@numba.njit(nogil=True, cache=True)
def jones_2x2_mul(ant1, bl, ant2, out):
    a2_xx_conj = np.conj(ant2[0, 0])
    a2_xy_conj = np.conj(ant2[0, 1])
    a2_yx_conj = np.conj(ant2[1, 0])
    a2_yy_conj = np.conj(ant2[1, 1])

    xx = bl[0, 0] * a2_xx_conj + bl[0, 1] * a2_yx_conj
    xy = bl[0, 0] * a2_xy_conj + bl[0, 1] * a2_yy_conj
    yx = bl[1, 0] * a2_xx_conj + bl[1, 1] * a2_yx_conj
    yy = bl[1, 0] * a2_xy_conj + bl[1, 1] * a2_yy_conj

    out[0, 0] = ant1[0, 0] * xx + ant1[0, 1] * yx
    out[0, 1] = ant1[0, 0] * xy + ant1[0, 1] * yy
    out[1, 0] = ant1[1, 0] * xx + ant1[1, 1] * yx
    out[1, 1] = ant1[1, 0] * xy + ant1[1, 1] * yy


@numba.njit(nogil=True)
def _predict_vis(time_index, ant1, ant2,
                 ant1_jones, ant2_jones, row_jones,
                 g1_jones, g2_jones, out, jones_mul):

    # Sanity check our indices
    for ti, a1, a2 in zip(time_index, ant1, ant2):
        if ti >= ant1_jones.shape[1] or ti >= ant2_jones.shape[1]:
            raise IndexError("Invalid time_index")

        if a1 >= ant1_jones.shape[2]:
            raise IndexError("Invalid antenna1 index")

        if a2 >= ant2_jones.shape[2]:
            raise IndexError("Invalid antenna2 index")

    jones_out = np.zeros(out.shape[2:], out.dtype)

    # Iterate over sources
    for s in range(ant1_jones.shape[0]):
        # Iterate over rows
        for r, (ti, a1, a2) in enumerate(zip(time_index, ant1, ant2)):
            # Iterate over channels
            for c in range(ant1_jones.shape[3]):
                jones_mul(ant1_jones[s, ti, a1, c], row_jones[s, r, c],
                          ant2_jones[s, ti, a2, c], jones_out)

                # Accumulate
                out[r, c] += jones_out

    # Iterate over rows
    for r, (ti, a1, a2) in enumerate(zip(time_index, ant1, ant2)):
        # Iterate over channels
        for c in range(ant1_jones.shape[3]):
            jones_mul(g1_jones[ti, a1, c], out[r, c],
                      g2_jones[ti, a2, c], jones_out)

            # Write output of multiplication back
            out[r, c] = jones_out


def predict_vis(time_index, antenna1, antenna2,
                ant1_jones, ant2_jones, row_jones,
                g1_jones, g2_jones):
    _, row, chan = row_jones.shape[:3]
    corrs = row_jones.shape[3:]

    if corrs == (1,) or corrs == (2,):
        jones_mul = jones_1_and_2_mul
    elif corrs == (2, 2):
        jones_mul = jones_2x2_mul
    else:
        raise ValueError("Unhandled correlations %s", (corrs,))

    out = np.zeros((row, chan) + corrs, dtype=row_jones.dtype)

    _predict_vis(time_index, antenna1, antenna2,
                 ant1_jones, ant2_jones, row_jones,
                 g1_jones, g2_jones, out, jones_mul)

    return out


_MP_DOCSTRING = namedtuple("MULTIPLEXDOCSTRING",
                           ["preamble", "notes", "parameters", "returns"])


predict_vis_docs = _MP_DOCSTRING(
    preamble="""
    Multiply Jones terms together to form model visibilities according
    to the following formula:

    .. math::

        V_{pq} = G_{p} \\left(
            \\sum_{s} A_{ps} B_{pqs} A_{qs}^H
            \\right) G_{q}^H

    where for antenna :math:`p` and :math:`q`, and source :math:`s`:

    - :math:`E_{ps}` represents direction-dependent (per-source) Jones terms.
    - :math:`B_{pqs}` represents a coherency matrix.
    - :math:`G_{p}` represents direction-independent Jones terms.

    Generally, :math:`E_{ps}` and :math:`G_{p}` should be formed by creating
    Jones terms using the `RIME API <rime-api-anchor_>`_ functions
    and combining them together with :func:`~numpy.einsum`.

    **Please read the Notes**

    """,

    notes="""
    Notes
    -----

    * The inputs to this function involve ``row``, ``time``
      and ``ant`` (antenna) dimensions.
    * Each ``row`` is associated with a pair of antenna Jones matrices
      at a particular timestep via the
      ``time_index``, ``antenna1`` and ``antenna2`` inputs.
    * The ``row`` dimension must be an increasing partial order in time.
    """,

    parameters="""
    Parameters
    ----------
    time_index : :class:`numpy.ndarray`
        Time index used to look up the antenna Jones index
        for a particular row (baseline).
        shape :code:`(row,)`.
    antenna1 : :class:`numpy.ndarray`
        Antenna 1 index used to look up the antenna Jones
        for a particular row (baseline).
        with shape :code:`(row,)`.
    antenna2 : :class:`numpy.ndarray`
        Antenna 2 index used to look up the antenna Jones
        for a particular row (baseline).
        with shape :code:`(row,)`.
    ant1_jones : :class:`numpy.ndarray`
        Per-source Jones terms for the first antenna.
        shape :code:`(source,time,ant,chan,corr_1,corr_2)`
    ant2_jones : :class:`numpy.ndarray`
        Per-source Jones terms for the second antenna.
        shape :code:`(source,time,ant,chan,corr_1,corr_2)`
    row_jones : :class:`numpy.ndarray`
        Per-source Jones term for the row (baseline).
        shape :code:`(source,row,chan,corr_1,corr_2)`
    g1_jones : :class:`numpy.ndarray`
        Jones terms for the first antenna of the baseline.
        shape :code:`(time,ant,chan,corr_1,corr_2)`
    g2_jones : :class:`numpy.ndarray`
        Jones terms for the second antenna of the baseline.
        shape :code:`(time,ant,chan,corr_1,corr_2)`
    """,

    returns="""
    Returns
    -------
    :class:`numpy.ndarray`
        Model visibilities of shape :code:`(row,chan,corr_1,corr_2)`
    """)


predict_vis.__doc__ = doc_tuple_to_str(predict_vis_docs)
