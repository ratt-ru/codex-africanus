# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import wraps

from ..compatibility import range

from .stokes_conversion import (stokes_convert_setup, stokes_convert_impl,
                                stokes_convert as np_stokes_convert,
                                STOKES_DOCS)

from ..util.docs import on_rtd
from ..util.requirements import have_packages, MissingPackageException

_package_requirements = ('dask.array',)
have_requirements = have_packages(*_package_requirements)


if not have_requirements or on_rtd():
    def stokes_convert(input, input_schema, output_schema):
        raise MissingPackageException(*_package_requirements)
else:
    import dask.array as da

    # This wraps is a https://en.wikipedia.org/wiki/Noble_lie
    @wraps(np_stokes_convert)
    def _wrapper(np_input, mapping=None, in_shape=None,
                 out_shape=None, dtype_=None):
        result = stokes_convert_impl(np_input, mapping, in_shape,
                                     out_shape, dtype_)

        # Introduce extra singleton dimension at the end of our shape
        return result.reshape(result.shape + (1,) * len(in_shape))

    def stokes_convert(input, input_schema, output_schema):
        mapping, in_shape, out_shape, dtype = stokes_convert_setup(
                                                    input,
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
        res = da.core.atop(_wrapper, free_dims + out_corr_dims + in_corr_dims,
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
    stokes_convert.__doc__ = (STOKES_DOCS
                              .format(array_type=":class:`dask.array.Array`"))
except AttributeError:
    pass
