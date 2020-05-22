# -*- coding: utf-8 -*-

import numpy as np

from africanus.util.numba import is_numba_type_none, generated_jit, njit, overload


def shape_or_invalid_shape(array, ndim):
    pass


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


@njit(nogil=True, inline='always')
def flags_match(flag_row, ri, out_flag_row, ro):
    if flag_row is None:
        return True
    else:
        return flag_row[ri] == out_flag_row[ro]


@njit(nogil=True, inline='always')
def is_chan_flagged(flag, r, f, c):
    return False if flag is None else flag[r, f, c]


@njit(nogil=True, inline='always')
def chan_add(output, input, orow, ochan, irow, ichan, corr):
    if input is not None:
        output[orow, ochan, corr] += input[irow, ichan, corr]


def vis_add(out_vis, out_weight_sum, in_vis,
             weight, weight_spectrum,
             orow, ochan, irow, ichan, corr):
    pass

@overload(vis_add, inline='always')
def _vis_add(out_vis, out_weight_sum, in_vis,
             weight, weight_spectrum,
             orow, ochan, irow, ichan, corr):
    """ Returns function adding weighted visibilities to a bin """
    if is_numba_type_none(in_vis):
        def impl(out_vis, out_weight_sum, in_vis,
                 weight, weight_spectrum,
                 orow, ochan, irow, ichan, corr):

            pass
    elif not is_numba_type_none(weight_spectrum):
        # Always prefer more accurate weight spectrum if we have it

        def impl(out_vis, out_weight_sum, in_vis,
                 weight, weight_spectrum,
                 orow, ochan, irow, ichan, corr):

            wt = weight_spectrum[irow, ichan, corr]
            iv = in_vis[irow, ichan, corr] * wt
            out_vis[orow, ochan, corr] += iv
            out_weight_sum[orow, ochan, corr] += wt
    elif not is_numba_type_none(weight):
        # Otherwise fall back to row weights
        def impl(out_vis, out_weight_sum, in_vis,
                 weight, weight_spectrum,
                 orow, ochan, irow, ichan, corr):

            wt = weight[irow, corr]
            iv = in_vis[irow, ichan, corr] * wt
            out_vis[orow, ochan, corr] += iv
            out_weight_sum[orow, ochan, corr] += wt
    else:
        # Natural weights
        def impl(out_vis, out_weight_sum, in_vis,
                 weight, weight_spectrum,
                 orow, ochan, irow, ichan, corr):

            iv = in_vis[irow, ichan, corr]
            out_vis[orow, ochan, corr] += iv
            out_weight_sum[orow, ochan, corr] += 1.0

    return impl

def sigma_spectrum_add(out_sigma, out_weight_sum, in_sigma,
                       weight, weight_spectrum,
                       orow, ochan, irow, ichan, corr):
    pass

@overload(sigma_spectrum_add, inline="always")
def _sigma_spectrum_add(out_sigma, out_weight_sum, in_sigma,
                        weight, weight_spectrum,
                        orow, ochan, irow, ichan, corr):
    """ Returns function adding weighted sigma to a bin """
    if is_numba_type_none(in_sigma):
        def impl(out_sigma, out_weight_sum, in_sigma,
                 weight, weight_spectrum,
                 orow, ochan, irow, ichan, corr):
            pass
    elif not is_numba_type_none(weight_spectrum):
        def impl(out_sigma, out_weight_sum, in_sigma,
                 weight, weight_spectrum,
                 orow, ochan, irow, ichan, corr):

            # Always prefer more accurate weight spectrum if we have it
            # sum(sigma**2 * weight**2)
            wt = weight_spectrum[irow, ichan, corr]
            is_ = in_sigma[irow, ichan, corr]**2 * wt**2
            out_sigma[orow, ochan, corr] += is_
            out_weight_sum[orow, ochan, corr] += wt

    elif not is_numba_type_none(weight):
        def impl(out_sigma, out_weight_sum, in_sigma,
                 weight, weight_spectrum,
                 orow, ochan, irow, ichan, corr):

            # sum(sigma**2 * weight**2)
            wt = weight[irow, corr]
            is_ = in_sigma[irow, ichan, corr]**2 * wt**2
            out_sigma[orow, ochan, corr] += is_
            out_weight_sum[orow, ochan, corr] += wt
    else:
        # Natural weights
        # sum(sigma**2 * weight**2)

        def impl(out_sigma, out_weight_sum, in_sigma,
                 weight, weight_spectrum,
                 orow, ochan, irow, ichan, corr):

            out_sigma[orow, ochan, corr] += in_sigma[irow, ichan, corr]**2
            out_weight_sum[orow, ochan, corr] += 1.0

    return impl

def normalise_vis(vis_out, vis_in, row, chan, corr, weight_sum):
    pass


@overload(normalise_vis, inline='always')
def _normalise_vis(vis_out, vis_in, row, chan, corr, weight_sum):
    if is_numba_type_none(vis_in):
        def impl(vis_out, vis_in, row, chan, corr, weight_sum):
            pass
    else:
        def impl(vis_out, vis_in, row, chan, corr, weight_sum):
            wsum = weight_sum[row, chan, corr]

            if wsum != 0.0:
                vis_out[row, chan, corr] = vis_in[row, chan, corr] / wsum
    return impl


def normalise_sigma_spectrum(sigma_out, sigma_in, row, chan, corr, weight_sum):
    pass


@overload(normalise_sigma_spectrum, inline='always')
def _normalise_sigma_spectrum(sigma_out, sigma_in, row, chan, corr, weight_sum):
    if is_numba_type_none(sigma_in) or is_numba_type_none(weight_sum):
        def impl(sigma_out, sigma_in, row, chan, corr, weight_sum):
            pass
    else:
        def impl(sigma_out, sigma_in, row, chan, corr, weight_sum):
            wsum = weight_sum[row, chan, corr]

            if wsum == 0.0:
                return

            # sqrt(sigma**2 * weight**2 / (weight(sum**2)))
            res = np.sqrt(sigma_in[row, chan, corr] / (wsum**2))
            sigma_out[row, chan, corr] = res

    return impl


def normalise_weight_spectrum(wt_spec_out, wt_spec_in, row, chan, corr):
    pass


@overload(normalise_weight_spectrum, inline='always')
def _normalise_weight_spectrum(wt_spec_out, wt_spec_in, row, chan, corr):
    if is_numba_type_none(wt_spec_in):
        def impl(wt_spec_out, wt_spec_in, row, chan, corr):
            pass
    else:
        def impl(wt_spec_out, wt_spec_in, row, chan, corr):
            wt_spec_out[row, chan, corr] = wt_spec_in[row, chan, corr]

    return impl