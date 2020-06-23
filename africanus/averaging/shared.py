# -*- coding: utf-8 -*-

import numpy as np

from africanus.util.numba import (is_numba_type_none,
                                  njit, overload,
                                  generated_jit)


def shape_or_invalid_shape(array, ndim):
    pass


# TODO(sjperkins)
# maybe replace with njit and inline='always' if
# https://github.com/numba/numba/issues/4693 is resolved
@generated_jit(nopython=True, nogil=True, cache=True)
def merge_flags(flag_row, flag):
    have_flag_row = not is_numba_type_none(flag_row)
    have_flag = not is_numba_type_none(flag)

    if have_flag_row and have_flag:
        def impl(flag_row, flag):
            """ Check flag_row and flag agree """
            for r in range(flag.shape[0]):
                all_flagged = True

                for f in range(flag.shape[1]):
                    for c in range(flag.shape[2]):
                        if flag[r, f, c] == 0:
                            all_flagged = False
                            break

                    if not all_flagged:
                        break

                if (flag_row[r] != 0) != all_flagged:
                    raise ValueError("flag_row and flag arrays mismatch")

            return flag_row

    elif have_flag_row and not have_flag:
        def impl(flag_row, flag):
            """ Return flag_row """
            return flag_row

    elif not have_flag_row and have_flag:
        def impl(flag_row, flag):
            """ Construct flag_row from flag """
            new_flag_row = np.empty(flag.shape[0], dtype=flag.dtype)

            for r in range(flag.shape[0]):
                all_flagged = True

                for f in range(flag.shape[1]):
                    for c in range(flag.shape[2]):
                        if flag[r, f, c] == 0:
                            all_flagged = False
                            break

                    if not all_flagged:
                        break

                new_flag_row[r] = (1 if all_flagged else 0)

            return new_flag_row

    else:
        def impl(flag_row, flag):
            return None

    return impl


@overload(shape_or_invalid_shape, inline='always')
def _shape_or_invalid_shape(array, ndim):
    """ Return array shape tuple or (-1,)*ndim if the array is None """

    try:
        ndim_lit = getattr(ndim, "literal_value")
    except AttributeError:
        raise ValueError("ndim must be a integer literal")

    if is_numba_type_none(array):
        tup = (-1,)*ndim_lit

        def impl(array, ndim):
            return tup
    else:
        def impl(array, ndim):
            return array.shape

    return impl


# TODO(sjperkins)
# maybe inline='always' if
# https://github.com/numba/numba/issues/4693 is resolved
@njit(nogil=True, cache=True)
def find_chan_corr(chan, corr, shape, chan_idx, corr_idx):
    """
    1. Get channel and correlation from shape if not set and the shape is valid
    2. Check they agree if they already agree

    Parameters
    ----------
    chan : int
        Existing channel size
    corr : int
        Existing correlation size
    shape : tuple
        Array shape tuple
    chan_idx : int
        Index of channel dimension in ``shape``.
    corr_idx : int
        Index of correlation dimension in ``shape``.

    Returns
    -------
    int
        Modified channel size
    int
        Modified correlation size
    """
    if chan_idx != -1:
        array_chan = shape[chan_idx]

        # Corresponds to a None array, ignore
        if array_chan == -1:
            pass
        # chan is not yet set, assign
        elif chan == 0:
            chan = array_chan
        # Check consistency
        elif chan != array_chan:
            raise ValueError("Inconsistent Channel Dimension "
                             "in Input Arrays")

    if corr_idx != -1:
        array_corr = shape[corr_idx]

        # Corresponds to a None array, ignore
        if array_corr == -1:
            pass
        # corr is not yet set, assign
        elif corr == 0:
            corr = array_corr
        # Check consistency
        elif corr != array_corr:
            raise ValueError("Inconsistent Correlation Dimension "
                             "in Input Arrays")

    return chan, corr


# TODO(sjperkins)
# maybe inline='always' if
# https://github.com/numba/numba/issues/4693 is resolved
@njit(nogil=True, cache=True)
def chan_corrs(vis, flag,
               weight_spectrum, sigma_spectrum,
               chan_freq, chan_width,
               effective_bw, resolution):
    """
    Infer channel and correlation size from input dimensions

    Returns
    -------
    int
        channel size
    int
        correlation size
    """
    vis_shape = shape_or_invalid_shape(vis, 3)
    flag_shape = shape_or_invalid_shape(flag, 3)
    weight_spectrum_shape = shape_or_invalid_shape(weight_spectrum, 3)
    sigma_spectrum_shape = shape_or_invalid_shape(sigma_spectrum, 3)
    chan_freq_shape = shape_or_invalid_shape(chan_freq, 1)
    chan_width_shape = shape_or_invalid_shape(chan_width, 1)
    effective_bw_shape = shape_or_invalid_shape(effective_bw, 1)
    resolution_shape = shape_or_invalid_shape(resolution, 1)

    chan = 0
    corr = 0

    chan, corr = find_chan_corr(chan, corr, vis_shape, 1, 2)
    chan, corr = find_chan_corr(chan, corr, flag_shape, 1, 2)
    chan, corr = find_chan_corr(chan, corr, weight_spectrum_shape, 1, 2)
    chan, corr = find_chan_corr(chan, corr, sigma_spectrum_shape, 1, 2)
    chan, corr = find_chan_corr(chan, corr, chan_freq_shape, 0, -1)
    chan, corr = find_chan_corr(chan, corr, chan_width_shape, 0, -1)
    chan, corr = find_chan_corr(chan, corr, effective_bw_shape, 0, -1)
    chan, corr = find_chan_corr(chan, corr, resolution_shape, 0, -1)

    return chan, corr


def flags_match(flag_row, ri, out_flag_row, ro):
    pass


@overload(flags_match, inline='always')
def _flags_match(flag_row, ri, out_flag_row, ro):
    if is_numba_type_none(flag_row):
        def impl(flag_row, ri, out_flag_row, ro):
            return True
    else:
        def impl(flag_row, ri, out_flag_row, ro):
            return flag_row[ri] == out_flag_row[ro]

    return impl
