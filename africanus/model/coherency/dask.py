# -*- coding: utf-8 -*-


from africanus.model.coherency.conversion import (convert_setup,
                                                  convert_impl,
                                                  CONVERT_DOCS)

from africanus.util.requirements import requires_optional

try:
    import dask.array as da
except ImportError as e:
    da_import_error = e
else:
    da_import_error = None


def convert_wrapper(np_input, mapping=None, in_shape=None,
                    out_shape=None, dtype_=None):
    result = convert_impl(np_input, mapping, in_shape,
                          out_shape, dtype_)

    # Introduce extra singleton dimension at the end of our shape
    return result.reshape(result.shape + (1,) * len(in_shape))


@requires_optional("dask.array", da_import_error)
def convert(input, input_schema, output_schema):
    mapping, in_shape, out_shape, dtype = convert_setup(input,
                                                        input_schema,
                                                        output_schema)

    n_free_dims = len(input.shape) - len(in_shape)
    free_dims = tuple("dim-%d" % i for i in range(n_free_dims))
    in_corr_dims = tuple("icorr-%d" % i for i in range(len(in_shape)))
    out_corr_dims = tuple("ocorr-%d" % i for i in range(len(out_shape)))

    # Output dimension are new dimensions
    new_axes = {d: s for d, s in zip(out_corr_dims, out_shape)}

    # Note the dummy in_corr_dims introduced at the end of our output,
    # We do this to prevent a contraction over the input dimensions
    # (which can be arbitrary) within the wrapper class
    res = da.core.blockwise(convert_wrapper,
                            free_dims + out_corr_dims + in_corr_dims,
                            input, free_dims + in_corr_dims,
                            mapping=mapping,
                            in_shape=in_shape,
                            out_shape=out_shape,
                            new_axes=new_axes,
                            dtype_=dtype,
                            dtype=dtype)

    # Now contract over the dummy dimensions
    start = len(free_dims) + len(out_corr_dims)
    end = start + len(in_corr_dims)
    return res.sum(axis=list(range(start, end)))


try:
    convert.__doc__ = CONVERT_DOCS.substitute(
                                array_type=":class:`dask.array.Array`")
except AttributeError:
    pass
