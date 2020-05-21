# -*- coding: utf-8 -*-

from africanus.util.numba import generated_jit, njit

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
def chan_adder(output, input, orow, ochan, irow, ichan, corr):
    if input is not None:
        output[orow, ochan, corr] += input[irow, ichan, corr]


@njit(nogil=True, inline='always')
def vis_add(out_vis, out_weight_sum, in_vis,
            weight, weight_spectrum,
            orow, ochan, irow, ichan, corr):
    """ Returns function adding weighted visibilities to a bin """
    if in_vis is None:
        pass
    elif weight_spectrum is not None:
        # Always prefer more accurate weight spectrum if we have it
        wt = weight_spectrum[irow, ichan, corr]
        iv = in_vis[irow, ichan, corr] * wt
        out_vis[orow, ochan, corr] += iv
        out_weight_sum[orow, ochan, corr] += wt
    elif weight is not None:
        # Otherwise fall back to row weights
        wt = weight[irow, corr]
        iv = in_vis[irow, ichan, corr] * wt
        out_vis[orow, ochan, corr] += iv
        out_weight_sum[orow, ochan, corr] += wt
    else:
        # Natural weights
        iv = in_vis[irow, ichan, corr]
        out_vis[orow, ochan, corr] += iv
        out_weight_sum[orow, ochan, corr] += 1.0


@njit(nogil=True, inline='always')
def sigma_spectrum_add(out_sigma, out_weight_sum, in_sigma,
                       weight, weight_spectrum,
                       orow, ochan, irow, ichan, corr):
    """ Returns function adding weighted sigma to a bin """
    if in_sigma is None:
        pass
    elif weight_spectrum is not None:
        # Always prefer more accurate weight spectrum if we have it
        # sum(sigma**2 * weight**2)
        wt = weight_spectrum[irow, ichan, corr]
        is_ = in_sigma[irow, ichan, corr]**2 * wt**2
        out_sigma[orow, ochan, corr] += is_
        out_weight_sum[orow, ochan, corr] += wt

    elif weight is not None:
        # sum(sigma**2 * weight**2)
        wt = weight[irow, corr]
        is_ = in_sigma[irow, ichan, corr]**2 * wt**2
        out_sigma[orow, ochan, corr] += is_
        out_weight_sum[orow, ochan, corr] += wt
    else:
        # Natural weights
        # sum(sigma**2 * weight**2)
        out_sigma[orow, ochan, corr] += in_sigma[irow, ichan, corr]**2
        out_weight_sum[orow, ochan, corr] += 1.0



@njit(nogil=True, inline='always')
def normalise_vis(vis_out, vis_in, row, chan, corr, weight_sum):
    if vis_in is not None:
        wsum = weight_sum[row, chan, corr]

        if wsum != 0.0:
            vis_out[row, chan, corr] = vis_in[row, chan, corr] / wsum


@njit(nogil=True, inline='always')
def normalise_sigma_spectrum(sigma_out, sigma_in, row, chan, corr, weight_sum):
    if sigma_in is not None:
        wsum = weight_sum[row, chan, corr]

        if wsum == 0.0:
            return

        # sqrt(sigma**2 * weight**2 / (weight(sum**2)))
        res = np.sqrt(sigma_in[row, chan, corr] / (wsum**2))
        sigma_out[row, chan, corr] = res

@njit(nogil=True, inline='always')
def normalise_weight_spectrum(wt_spec_out, wt_spec_in, row, chan, corr):
    if wt_spec_in is not None:
        wt_spec_out[row, chan, corr] = wt_spec_in[row, chan, corr]
