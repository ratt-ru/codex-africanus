# -*- coding: utf-8 -*-

import numpy as np

from africanus.util.numba import (is_numba_type_none,
                                  intrinsic,
                                  njit,
                                  generated_jit,
                                  overload)


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

    import numba.core.types as nbtypes
    from numba.extending import SentryLiteralArgs

    SentryLiteralArgs(['ndim']).for_function(
        _shape_or_invalid_shape).bind(array, ndim)

    try:
        ndim_lit = getattr(ndim, "literal_value")
    except AttributeError:
        raise ValueError("ndim must be a integer literal")

    if is_numba_type_none(array):
        tup = (-1,)*ndim_lit

        def impl(array, ndim):
            return tup

        return impl
    elif isinstance(array, nbtypes.Array):
        def impl(array, ndim):
            return array.shape

        return impl
    elif (isinstance(array, nbtypes.UniTuple) and
            isinstance(array.dtype, nbtypes.Array)):

        if len(array) == 1:
            def impl(array, ndim):
                return array[0].shape
        else:
            def impl(array, ndim):
                shape = array[0].shape

                for a in array[1:]:
                    if a.shape != shape:
                        raise ValueError("Array shapes in Tuple don't match")

                return shape

        return impl
    elif isinstance(array, nbtypes.Tuple):
        if not all(isinstance(a, nbtypes.Array) for a in array.types):
            raise ValueError("Must be Tuple of Arrays")

        if not all(array.types[0].ndim == a.ndim for a in array.types[1:]):
            raise ValueError("Array ndims in Tuple don't match")

        if len(array) == 1:
            def impl(array, ndim):
                return array[0].shape
        else:
            def impl(array, ndim):
                shape = array[0].shape

                for a in array[1:]:
                    if a.shape != shape:
                        raise ValueError("Array shapes in Tuple don't match")

                return shape

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


@intrinsic
def vis_output_arrays(typingctx, vis, out_shape):
    from numba.core import types, cgutils
    from numba.np import numpy_support

    def vis_weight_types(vis):
        """ Determine output visibility and weight types """

        if isinstance(vis.dtype, types.Complex):
            # Use the float representation as dtype if vis is complex
            # (and it will be most of the time)
            weight_type = vis.dtype.underlying_float
        else:
            # Default case
            weight_type = vis.dtype

        ndim = out_shape.count
        # Visibility type will be the same, except for ndim change
        avg_array_type = vis.copy(ndim=ndim)
        # Weight type changes the dtype and ndim
        weight_array_type = vis.copy(ndim=ndim, dtype=weight_type)

        return avg_array_type, weight_array_type

    if isinstance(vis, types.Array):
        have_vis_array = True
        have_vis_tuple = False

        vt, wt = vis_weight_types(vis)
        return_type = types.Tuple((vt, wt))
    elif isinstance(vis, types.UniTuple):
        have_vis_array = False
        have_vis_tuple = True

        vt, wt = vis_weight_types(vis.dtype)
        vt = (vt,) * vis.count
        wt = (wt,) * vis.count
        # Create a two-tier tuple (likely heterogenous):
        # (avg_vis, avg_vis_weights))
        return_type = types.Tuple(tuple(map(types.Tuple, (vt, wt))))

    elif isinstance(vis, types.Tuple):
        have_vis_array = False
        have_vis_tuple = True

        vt, wt = zip(*(vis_weight_types(v) for v in vis))
        # Create a two-tier tuple (likely heterogenous):
        # (avg_vis, avg_vis_weights))
        return_type = types.Tuple(tuple(map(types.Tuple, (vt, wt))))
    else:
        raise TypeError(f"vis must be an Array or Tuple. Got {vis}")

    sig = return_type(vis, out_shape)

    def codegen(context, builder, signature, args):
        vis_type, out_shape_type = signature.args
        return_type = signature.return_type
        vis, out_shape = args

        # Create the outer tuple
        llvm_outer_tuple_type = context.get_value_type(return_type)
        outer_tuple = cgutils.get_null_value(llvm_outer_tuple_type)

        def gen_array_factory(numba_dtype):
            """
            Create a funtion that creates an array.
            Bind the numpy dtype because I don't know how to create
            the numba version
            """
            np_dtype = numpy_support.as_dtype(numba_dtype)
            return lambda shape: np.zeros(shape, np_dtype)

        for i, inner_type in enumerate(return_type.types):
            # Generate an array and insert into return tuple
            if have_vis_array:
                array_factory = gen_array_factory(inner_type.dtype)
                factory_sig = inner_type(out_shape_type)
                factory_args = [out_shape]

                # Compile function and get handle to output array
                inner_value = context.compile_internal(builder,
                                                       array_factory,
                                                       factory_sig,
                                                       factory_args)

            # Insert inner tuple into outer tuple
            elif have_vis_tuple:
                # Create the inner tuple
                llvm_inner_type = context.get_value_type(inner_type)
                inner_value = cgutils.get_null_value(llvm_inner_type)

                for j, array_type in enumerate(inner_type.types):
                    # Create function, it's signature and arguments
                    array_factory = gen_array_factory(array_type.dtype)
                    factory_sig = array_type(out_shape_type)
                    factory_args = [out_shape]

                    # Compile function and get handle to output
                    data = context.compile_internal(builder, array_factory,
                                                    factory_sig, factory_args)

                    # Insert data into inner_value
                    inner_value = builder.insert_value(inner_value, data, j)
            else:
                raise ValueError("Internal logic error")

            # Insert inner tuple into outer tuple
            outer_tuple = builder.insert_value(outer_tuple, inner_value, i)

        return outer_tuple

    return sig, codegen
