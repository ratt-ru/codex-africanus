# -*- coding: utf-8 -*-


import numpy as np

from africanus.util.docs import DocstringTemplate
from africanus.util.numba import is_numba_type_none, generated_jit, njit


JONES_NOT_PRESENT = 0
JONES_1_OR_2 = 1
JONES_2X2 = 2


def _get_jones_types(name, numba_ndarray_type, corr_1_dims, corr_2_dims):
    """
    Determine which of the following three cases are valid:

    1. The array is not present (None) and therefore no Jones Matrices
    2. single (1,) or (2,) dual correlation
    3. (2, 2) full correlation

    Parameters
    ----------
    name: str
        Array name
    numba_ndarray_type: numba.type
        Array numba type
    corr_1_dims: int
        Number of `numba_ndarray_type` dimensions,
        including correlations (first option)
    corr_2_dims: int
        Number of `numba_ndarray_type` dimensions,
        including correlations (second option)

    Returns
    -------
    int
        Enumeration describing the Jones Matrix Type

        - 0 -- Not Present
        - 1 -- (1,) or (2,)
        - 2 -- (2, 2)
    """

    if is_numba_type_none(numba_ndarray_type):
        return JONES_NOT_PRESENT
    if numba_ndarray_type.ndim == corr_1_dims:
        return JONES_1_OR_2
    elif numba_ndarray_type.ndim == corr_2_dims:
        return JONES_2X2
    else:
        raise ValueError("%s.ndim not in (%d, %d)" %
                         (name, corr_1_dims, corr_2_dims))


def jones_mul_factory(have_ddes, have_coh, jones_type, accumulate):
    """
    Outputs a function that multiplies some combination of
    (dde1_jones, baseline_jones, dde2_jones) together.

    Parameters
    ----------
    have_ddes : boolean
        If True, indicates that antenna jones terms are present
    have_coh : boolean
        If True, indicates that baseline jones terms are present
    jones_type : int
        Type of Jones matrix
    accumulate : boolean
        If True, the result of the multiplication is accumulated
        into the output, otherwise, it is assigned

    Notes
    -----
    ``accumulate`` is treated by LLVM as a compile-time constant,
    according to https://numba.pydata.org/numba-doc/latest/glossary.html.

    Therefore in principle, the conditional checks
    involving ``accumulate`` inside the functions should
    be elided by the compiler.


    Returns
    -------
    callable
        jitted numba function performing the Jones Multiply
    """
    ex = ValueError("Invalid Jones Type %s" % jones_type)

    if have_coh and have_ddes:
        if jones_type == JONES_1_OR_2:
            def jones_mul(a1j, blj, a2j, jout):
                for c in range(jout.shape[0]):
                    if accumulate:
                        jout[c] += a1j[c] * blj[c] * np.conj(a2j[c])
                    else:
                        jout[c] = a1j[c] * blj[c] * np.conj(a2j[c])

        elif jones_type == JONES_2X2:
            def jones_mul(a1j, blj, a2j, jout):
                a2_xx_H = np.conj(a2j[0, 0])
                a2_xy_H = np.conj(a2j[0, 1])
                a2_yx_H = np.conj(a2j[1, 0])
                a2_yy_H = np.conj(a2j[1, 1])

                xx = blj[0, 0] * a2_xx_H + blj[0, 1] * a2_xy_H
                xy = blj[0, 0] * a2_yx_H + blj[0, 1] * a2_yy_H
                yx = blj[1, 0] * a2_xx_H + blj[1, 1] * a2_xy_H
                yy = blj[1, 0] * a2_yx_H + blj[1, 1] * a2_yy_H

                if accumulate:
                    jout[0, 0] += a1j[0, 0] * xx + a1j[0, 1] * yx
                    jout[0, 1] += a1j[0, 0] * xy + a1j[0, 1] * yy
                    jout[1, 0] += a1j[1, 0] * xx + a1j[1, 1] * yx
                    jout[1, 1] += a1j[1, 0] * xy + a1j[1, 1] * yy
                else:
                    jout[0, 0] = a1j[0, 0] * xx + a1j[0, 1] * yx
                    jout[0, 1] = a1j[0, 0] * xy + a1j[0, 1] * yy
                    jout[1, 0] = a1j[1, 0] * xx + a1j[1, 1] * yx
                    jout[1, 1] = a1j[1, 0] * xy + a1j[1, 1] * yy

        else:
            raise ex
    elif have_ddes and not have_coh:
        if jones_type == JONES_1_OR_2:
            def jones_mul(a1j, a2j, jout):
                for c in range(jout.shape[0]):
                    if accumulate:
                        jout[c] += a1j[c] * np.conj(a2j[c])
                    else:
                        jout[c] = a1j[c] * np.conj(a2j[c])

        elif jones_type == JONES_2X2:
            def jones_mul(a1j, a2j, jout):
                a2_xx_H = np.conj(a2j[0, 0])
                a2_xy_H = np.conj(a2j[0, 1])
                a2_yx_H = np.conj(a2j[1, 0])
                a2_yy_H = np.conj(a2j[1, 1])

                if accumulate:
                    jout[0, 0] += a1j[0, 0] * a2_xx_H + a1j[0, 1] * a2_xy_H
                    jout[0, 1] += a1j[0, 0] * a2_yx_H + a1j[0, 1] * a2_yy_H
                    jout[1, 0] += a1j[1, 0] * a2_xx_H + a1j[1, 1] * a2_xy_H
                    jout[1, 1] += a1j[1, 0] * a2_yx_H + a1j[1, 1] * a2_yy_H
                else:
                    jout[0, 0] += a1j[0, 0] * a2_xx_H + a1j[0, 1] * a2_xy_H
                    jout[0, 1] += a1j[0, 0] * a2_yx_H + a1j[0, 1] * a2_yy_H
                    jout[1, 0] += a1j[1, 0] * a2_xx_H + a1j[1, 1] * a2_xy_H
                    jout[1, 1] += a1j[1, 0] * a2_yx_H + a1j[1, 1] * a2_yy_H
        else:
            raise ex
    elif not have_ddes and have_coh:
        if jones_type == JONES_1_OR_2:
            def jones_mul(blj, jout):
                for c in range(jout.shape[0]):
                    if accumulate:
                        jout[c] += blj[c]
                    elif id(blj) == id(jout):
                        pass
                    else:
                        jout[c] = blj[c]

        elif jones_type == JONES_2X2:
            def jones_mul(blj, jout):
                if accumulate:
                    jout[0, 0] += blj[0, 0]
                    jout[0, 1] += blj[0, 1]
                    jout[1, 0] += blj[1, 0]
                    jout[1, 1] += blj[1, 1]
                elif id(blj) == id(jout):
                    pass
                else:
                    jout[0, 0] = blj[0, 0]
                    jout[0, 1] = blj[0, 1]
                    jout[1, 0] = blj[1, 0]
                    jout[1, 1] = blj[1, 1]
        else:
            raise ex
    else:
        # noop
        def jones_mul():
            pass

    return njit(nogil=True, inline="always")(jones_mul)


def sum_coherencies_factory(have_ddes, have_coh, jones_type):
    """ Factory function generating a function that sums coherencies """
    jones_mul = jones_mul_factory(have_ddes, have_coh, jones_type, True)

    if have_ddes and have_coh:
        def sum_coh_fn(time, ant1, ant2, a1j, blj, a2j, tmin, cout):
            for s in range(a1j.shape[0]):
                for r in range(time.shape[0]):
                    ti = time[r] - tmin
                    a1 = ant1[r]
                    a2 = ant2[r]

                    for f in range(a1j.shape[3]):
                        jones_mul(a1j[s, ti, a1, f],
                                  blj[s, r, f],
                                  a2j[s, ti, a2, f],
                                  cout[r, f])

    elif have_ddes and not have_coh:
        def sum_coh_fn(time, ant1, ant2, a1j, blj, a2j, tmin, cout):
            for s in range(a1j.shape[0]):
                for r in range(time.shape[0]):
                    ti = time[r] - tmin
                    a1 = ant1[r]
                    a2 = ant2[r]

                    for f in range(a1j.shape[3]):
                        jones_mul(a1j[s, ti, a1, f],
                                  a2j[s, ti, a2, f],
                                  cout[r, f])

    elif not have_ddes and have_coh:
        if jones_type == JONES_2X2:
            def sum_coh_fn(time, ant1, ant2, a1j, blj, a2j, tmin, cout):
                for s in range(blj.shape[0]):
                    for r in range(blj.shape[1]):
                        for f in range(blj.shape[2]):
                            for c1 in range(blj.shape[3]):
                                for c2 in range(blj.shape[4]):
                                    cout[r, f, c1, c2] += blj[s, r, f, c1, c2]
        else:
            def sum_coh_fn(time, ant1, ant2, a1j, blj, a2j, tmin, cout):
                # TODO(sjperkins): Without this, these loops
                # produce an incorrect value
                assert blj.ndim == 4
                for s in range(blj.shape[0]):
                    for r in range(blj.shape[1]):
                        for f in range(blj.shape[2]):
                            for c in range(blj.shape[3]):
                                cout[r, f, c] += blj[s, r, f, c]
    else:
        # noop
        def sum_coh_fn(time, ant1, ant2, a1j, blj, a2j, tmin, cout):
            pass

    return njit(nogil=True, inline="always")(sum_coh_fn)


def output_factory(have_ddes, have_coh, have_dies, have_base_vis, out_dtype):
    """ Factory function generating a function that creates function output """
    if have_ddes:
        def output(time_index, dde1_jones, source_coh, dde2_jones,
                   die1_jones, base_vis, die2_jones):
            row = time_index.shape[0]
            chan = dde1_jones.shape[3]
            corrs = dde1_jones.shape[4:]
            return np.zeros((row, chan) + corrs, dtype=out_dtype)
    elif have_coh:
        def output(time_index, dde1_jones, source_coh, dde2_jones,
                   die1_jones, base_vis, die2_jones):
            row = time_index.shape[0]
            chan = source_coh.shape[2]
            corrs = source_coh.shape[3:]
            return np.zeros((row, chan) + corrs, dtype=out_dtype)
    elif have_dies:
        def output(time_index, dde1_jones, source_coh, dde2_jones,
                   die1_jones, base_vis, die2_jones):
            row = time_index.shape[0]
            chan = die1_jones.shape[2]
            corrs = die1_jones.shape[3:]
            return np.zeros((row, chan) + corrs, dtype=out_dtype)
    elif have_base_vis:
        def output(time_index, dde1_jones, source_coh, dde2_jones,
                   die1_jones, base_vis, die2_jones):
            row = time_index.shape[0]
            chan = base_vis.shape[1]
            corrs = base_vis.shape[2:]
            return np.zeros((row, chan) + corrs, dtype=out_dtype)

    else:
        raise ValueError("Insufficient inputs were supplied "
                         "for determining the output shape")

    # TODO(sjperkins)
    # perhaps inline="always" on resolution of
    # https://github.com/numba/numba/issues/4691
    return njit(nogil=True, inline='never')(output)


def add_coh_factory(have_bvis):
    if have_bvis:
        def add_coh(base_vis, add_coh_cout):
            add_coh_cout += base_vis
    else:
        # noop
        def add_coh(base_vis, add_coh_cout):
            pass

    return njit(nogil=True, inline="always")(add_coh)


def apply_dies_factory(have_dies, jones_type):
    """
    Factory function returning a function that applies
    Direction Independent Effects
    """

    # We always "have visibilities", (the output array)
    jones_mul = jones_mul_factory(have_dies, True, jones_type, False)

    if have_dies:
        def apply_dies(time, ant1, ant2,
                       die1_jones, die2_jones,
                       tmin, dies_out):
            # Iterate over rows
            for r in range(time.shape[0]):
                ti = time[r] - tmin
                a1 = ant1[r]
                a2 = ant2[r]

                # Iterate over channels
                for c in range(dies_out.shape[1]):
                    jones_mul(die1_jones[ti, a1, c], dies_out[r, c],
                              die2_jones[ti, a2, c], dies_out[r, c])
    else:
        # noop
        def apply_dies(time, ant1, ant2,
                       die1_jones, die2_jones,
                       tmin, dies_out):
            pass

    return njit(nogil=True, inline="always")(apply_dies)


def _default_none_check(arg):
    return arg is not None


def predict_checks(time_index, antenna1, antenna2,
                   dde1_jones, source_coh, dde2_jones,
                   die1_jones, base_vis, die2_jones,
                   none_check=_default_none_check):

    have_ddes1 = none_check(dde1_jones)
    have_coh = none_check(source_coh)
    have_ddes2 = none_check(dde2_jones)
    have_dies1 = none_check(die1_jones)
    have_bvis = none_check(base_vis)
    have_dies2 = none_check(die2_jones)

    assert time_index.ndim == 1
    assert antenna1.ndim == 1
    assert antenna2.ndim == 1

    if have_ddes1 ^ have_ddes2:
        raise ValueError("Both dde1_jones and dde2_jones "
                         "must be present or absent")

    if have_dies1 ^ have_dies2:
        raise ValueError("Both die1_jones and die2_jones "
                         "must be present or absent")

    have_ddes = have_ddes1 and have_ddes2
    have_dies = have_dies1 and have_dies2

    if have_ddes1 and dde1_jones.ndim not in (5, 6):
        raise ValueError("dde1_jones.ndim %d not in (5, 6)" % dde1_jones.ndim)

    if have_ddes2 and dde2_jones.ndim not in (5, 6):
        raise ValueError("dde2_jones.ndim %d not in (5, 6)" % dde2_jones.ndim)

    if have_ddes and dde1_jones.ndim != dde2_jones.ndim:
        raise ValueError("dde1_jones.ndim != dde2_jones.ndim")

    if have_coh and source_coh.ndim not in (4, 5):
        raise ValueError("source_coh.ndim %d not in (4, 5)" % source_coh.ndim)

    if have_dies1 and die1_jones.ndim not in (4, 5):
        raise ValueError("die1_jones.ndim %d not in (4, 5)" % die1_jones.ndim)

    if have_bvis and base_vis.ndim not in (3, 4):
        raise ValueError("base_vis.ndim %d not in (3, 4)" % base_vis.ndim)

    if have_dies2 and die2_jones.ndim not in (4, 5):
        raise ValueError("die2_jones.ndim %d not in (4, 5)" % die2_jones.ndim)

    if have_dies1 and have_dies2 and die1_jones.ndim != die2_jones.ndim:
        raise ValueError("die1_jones.ndim != die2_jones.ndim")

    expected_sizes = []

    if have_ddes:
        ndim = dde1_jones.ndim
        expected_sizes.append([ndim, ndim - 1, ndim - 2, ndim - 1]),

    if have_coh:
        ndim = source_coh.ndim
        expected_sizes.append([ndim + 1, ndim, ndim - 1, ndim])

    if have_dies:
        ndim = die1_jones.ndim
        expected_sizes.append([ndim + 1, ndim, ndim - 1, ndim])

    if have_bvis:
        ndim = base_vis.ndim
        expected_sizes.append([ndim + 2, ndim + 1, ndim, ndim + 1])

    if not all(expected_sizes[0] == s for s in expected_sizes[1:]):
        raise ValueError("One of the following pre-conditions is broken "
                         "(missing values are ignored):\n"
                         "dde_jones{1,2}.ndim == source_coh.ndim + 1\n"
                         "dde_jones{1,2}.ndim == base_vis.ndim + 2\n"
                         "dde_jones{1,2}.ndim == die_jones{1,2}.ndim + 1")

    return (have_ddes1, have_coh, have_ddes2,
            have_dies1, have_bvis, have_dies2)


@generated_jit(nopython=True, nogil=True, cache=True)
def predict_vis(time_index, antenna1, antenna2,
                dde1_jones=None, source_coh=None, dde2_jones=None,
                die1_jones=None, base_vis=None, die2_jones=None):

    tup = predict_checks(time_index, antenna1, antenna2,
                         dde1_jones, source_coh, dde2_jones,
                         die1_jones, base_vis, die2_jones,
                         lambda x: not is_numba_type_none(x))

    (have_ddes1, have_coh, have_ddes2, have_dies1, have_bvis, have_dies2) = tup

    # Infer the output dtype
    dtype_arrays = (dde1_jones, source_coh, dde2_jones,
                    die1_jones, base_vis, die2_jones)

    out_dtype = np.result_type(*(np.dtype(a.dtype.name)
                                 for a in dtype_arrays
                                 if not is_numba_type_none(a)))

    jones_types = [
        _get_jones_types("dde1_jones", dde1_jones, 5, 6),
        _get_jones_types("source_coh", source_coh, 4, 5),
        _get_jones_types("dde2_jones", dde2_jones, 5, 6),
        _get_jones_types("die1_jones", die1_jones, 4, 5),
        _get_jones_types("base_vis", base_vis, 3, 4),
        _get_jones_types("die2_jones", die2_jones, 4, 5)]

    ptypes = [t for t in jones_types if t != JONES_NOT_PRESENT]

    if not all(ptypes[0] == p for p in ptypes[1:]):
        raise ValueError("Jones Matrix Correlations were mismatched")

    try:
        jones_type = ptypes[0]
    except IndexError:
        raise ValueError("No Jones Matrices were supplied")

    have_ddes = have_ddes1 and have_ddes2
    have_dies = have_dies1 and have_dies2

    # Create functions that we will use inside our predict function
    out_fn = output_factory(have_ddes, have_coh,
                            have_dies, have_bvis, out_dtype)
    sum_coh_fn = sum_coherencies_factory(have_ddes, have_coh, jones_type)
    apply_dies_fn = apply_dies_factory(have_dies, jones_type)
    add_coh_fn = add_coh_factory(have_bvis)

    def _predict_vis_fn(time_index, antenna1, antenna2,
                        dde1_jones=None, source_coh=None, dde2_jones=None,
                        die1_jones=None, base_vis=None, die2_jones=None):

        # Get the output shape
        out = out_fn(time_index, dde1_jones, source_coh, dde2_jones,
                     die1_jones, base_vis, die2_jones)

        # Minimum time index, used to normalise within function
        tmin = time_index.min()

        # Sum coherencies if any
        sum_coh_fn(time_index, antenna1, antenna2,
                   dde1_jones, source_coh, dde2_jones,
                   tmin, out)

        # Add base visibilities to the output, if any
        add_coh_fn(base_vis, out)

        # Apply direction independent effects, if any
        apply_dies_fn(time_index, antenna1, antenna2,
                      die1_jones, die2_jones,
                      tmin, out)

        return out

    return _predict_vis_fn


@generated_jit(nopython=True, nogil=True, cache=True)
def apply_gains(time_index, antenna1, antenna2,
                die1_jones, corrupted_vis, die2_jones):

    def impl(time_index, antenna1, antenna2,
             die1_jones, corrupted_vis, die2_jones):
        return predict_vis(time_index, antenna1, antenna2,
                           die1_jones=die1_jones,
                           base_vis=corrupted_vis,
                           die2_jones=die2_jones)

    return impl


PREDICT_DOCS = DocstringTemplate(r"""
Multiply Jones terms together to form model visibilities according
to the following formula:

.. math::


    V_{pq} = G_{p} \left(
        B_{pq} + \sum_{s} E_{ps} X_{pqs} E_{qs}^H
        \right) G_{q}^H

where for antenna :math:`p` and :math:`q`, and source :math:`s`:


- :math:`B_{{pq}}` represent base coherencies.
- :math:`E_{{ps}}` represents Direction-Dependent Jones terms.
- :math:`X_{{pqs}}` represents a coherency matrix (per-source).
- :math:`G_{{p}}` represents Direction-Independent Jones terms.

Generally, :math:`E_{ps}`, :math:`G_{p}`, :math:`X_{pqs}`
should be formed by using the `RIME API <rime-api-anchor_>`_ functions
and combining them together with :func:`~numpy.einsum`.

**Please read the Notes**

Notes
-----
* Direction-Dependent terms (dde{1,2}_jones) and
  Independent (die{1,2}_jones) are optional,
  but if one is present, the other must be present.
* The inputs to this function involve ``row``, ``time``
  and ``ant`` (antenna) dimensions.
* Each ``row`` is associated with a pair of antenna Jones matrices
  at a particular timestep via the
  ``time_index``, ``antenna1`` and ``antenna2`` inputs.
* The ``row`` dimension must be an increasing partial order in time.
$(extra_notes)


Parameters
----------
time_index : $(array_type)
    Time index used to look up the antenna Jones index
    for a particular baseline with shape :code:`(row,)`.
    Obtainable via $(get_time_index).
antenna1 : $(array_type)
    Antenna 1 index used to look up the antenna Jones
    for a particular baseline.
    with shape :code:`(row,)`.
antenna2 : $(array_type)
    Antenna 2 index used to look up the antenna Jones
    for a particular baseline.
    with shape :code:`(row,)`.
dde1_jones : $(array_type), optional
    :math:`E_{ps}` Direction-Dependent Jones terms for the first antenna.
    shape :code:`(source,time,ant,chan,corr_1,corr_2)`
source_coh : $(array_type), optional
    :math:`X_{pqs}` Direction-Dependent Coherency matrix for the baseline.
    with shape :code:`(source,row,chan,corr_1,corr_2)`
dde2_jones : $(array_type), optional
    :math:`E_{qs}` Direction-Dependent Jones terms for the second antenna.
    This is usually the same array as ``dde1_jones`` as this
    preserves the symmetry of the RIME. ``predict_vis`` will
    perform the conjugate transpose internally.
    shape :code:`(source,time,ant,chan,corr_1,corr_2)`
die1_jones : $(array_type), optional
    :math:`G_{ps}` Direction-Independent Jones terms for the
    first antenna of the baseline.
    with shape :code:`(time,ant,chan,corr_1,corr_2)`
base_vis : $(array_type), optional
    :math:`B_{pq}` base coherencies, added to source coherency summation
    *before* multiplication with `die1_jones` and `die2_jones`.
    shape :code:`(row,chan,corr_1,corr_2)`.
die2_jones : $(array_type), optional
    :math:`G_{ps}` Direction-Independent Jones terms for the
    second antenna of the baseline.
    This is usually the same array as ``die1_jones`` as this
    preserves the symmetry of the RIME. ``predict_vis`` will
    perform the conjugate transpose internally.
    shape :code:`(time,ant,chan,corr_1,corr_2)`
$(extra_args)

Returns
-------
visibilities : $(array_type)
    Model visibilities of shape :code:`(row,chan,corr_1,corr_2)`
""")


try:
    predict_vis.__doc__ = PREDICT_DOCS.substitute(
                            array_type=":class:`numpy.ndarray`",
                            get_time_index=":code:`np.unique(time, "
                                           "return_inverse=True)[1]`",
                            extra_args="",
                            extra_notes="")
except AttributeError:
    pass


APPLY_GAINS_DOCS = DocstringTemplate(r"""
Apply gains to corrupted visibilities in order to recover
the true visibilities.

Thin wrapper around $(wrapper_func).

.. math::


    V_{pq} = G_{p} C_{pq} G_{q}^H


Parameters
----------
time_index : $(array_type)
    Time index used to look up the antenna Jones index
    for a particular baseline
    with shape :code:`(row,)`.
antenna1 : $(array_type)
    Antenna 1 index used to look up the antenna Jones
    for a particular baseline
    with shape :code:`(row,)`.
antenna2 : $(array_type)
    Antenna 2 index used to look up the antenna Jones
    for a particular baseline
    with shape :code:`(row,)`.
gains1 : $(array_type), optional
    :math:`G_{ps}` Gains for the first antenna of the baseline.
    with shape :code:`(time,ant,chan,corr_1,corr_2)`
corrupted_vis : $(array_type), optiona
    :math:`B_{pq}` corrupted visibilities.
    with shape :code:`(row,chan,corr_1,corr_2)`.
gains2 : $(array_type), optional
    :math:`G_{ps}` Gains for the second antenna of the baseline
    with shape :code:`(time,ant,chan,corr_1,corr_2)`.

Returns
-------
true_vis : $(array_type)
    True visibilities of shape :code:`(row,chan,corr_1,corr_2)`
""")

try:
    apply_gains.__doc__ = APPLY_GAINS_DOCS.substitute(
                        array_type=":class:`numpy.ndarray`",
                        wrapper_func=":func:`~africanus.rime.predict_vis`")
except AttributeError:
    pass
