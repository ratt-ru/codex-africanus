# -*- coding: utf-8 -*-


from functools import reduce
import logging
from operator import mul
from os.path import join as pjoin

import numpy as np

from africanus.model.coherency.conversion import (_element_indices_and_shape,
                                                  CONVERT_DOCS,
                                                  MissingConversionInputs)
from africanus.util.code import memoize_on_key, format_code
from africanus.util.cuda import cuda_type, grids
from africanus.util.jinja2 import jinja_env
from africanus.util.requirements import requires_optional

try:
    import cupy as cp
    from cupy.cuda.compiler import CompileException
except ImportError as e:
    opt_import_error = e
else:
    opt_import_error = None

log = logging.getLogger(__name__)

stokes_conv = {
    'RR': {('I', 'V'): ("complex", "make_{{out_type}}({{I}} + {{V}}, 0)")},
    'RL': {('Q', 'U'): ("complex", "make_{{out_type}}({{Q}}, {{U}})")},
    'LR': {('Q', 'U'): ("complex", "make_{{out_type}}({{Q}}, -{{U}})")},
    'LL': {('I', 'V'): ("complex", "make_{{out_type}}({{I}} - {{V}}, 0)")},

    'XX': {('I', 'Q'): ("complex", "make_{{out_type}}({{I}} + {{Q}}, 0)")},
    'XY': {('U', 'V'): ("complex", "make_{{out_type}}({{U}}, {{V}})")},
    'YX': {('U', 'V'): ("complex", "make_{{out_type}}({{U}}, -{{V}})")},
    'YY': {('I', 'Q'): ("complex", "make_{{out_type}}({{I}} - {{Q}}, 0)")},

    'I': {('XX', 'YY'): ("real", "(({{XX}}.x + {{YY}}.x) / 2)"),
          ('RR', 'LL'): ("real", "(({{RR}}.x + {{LL}}.x) / 2)")},

    'Q': {('XX', 'YY'): ("real", "(({{XX}}.x - {{YY}}.x) / 2)"),
          ('RL', 'LR'): ("real", "(({{RL}}.x + {{LR}}.x) / 2)")},

    'U': {('XY', 'YX'): ("real", "(({{XY}}.x + {{YX}}.x) / 2)"),
          ('RL', 'LR'): ("real", "(({{RL}}.y - {{LR}}.y) / 2)")},

    'V': {('XY', 'YX'): ("real", "(({{XY}}.y - {{YX}}.y) / 2)"),
          ('RR', 'LL'): ("real", "(({{RR}}.x - {{LL}}.x) / 2)")},
}


def stokes_convert_setup(input, input_schema, output_schema):
    input_indices, input_shape = _element_indices_and_shape(input_schema)
    output_indices, output_shape = _element_indices_and_shape(output_schema)

    if input.shape[-len(input_shape):] != input_shape:
        raise ValueError("Last dimension of input doesn't match input schema")

    mapping = []
    dtypes = []

    # Figure out how to produce an output from available inputs
    for okey, out_idx in output_indices.items():
        try:
            deps = stokes_conv[okey]
        except KeyError:
            raise ValueError("Unknown output '%s'. Known types '%s'"
                             % (okey, stokes_conv.keys()))

        found_conv = False

        # Find a mapping for which we have inputs
        for (c1, c2), (dtype, fn) in deps.items():
            # Get indices for both correlations
            try:
                c1_idx = input_indices[c1]
            except KeyError:
                continue

            try:
                c2_idx = input_indices[c2]
            except KeyError:
                continue

            found_conv = True
            dtypes.append(dtype)

            mapping.append(((c1, c1_idx), (c2, c2_idx), out_idx, fn))
            break

        # We must find a conversion
        if not found_conv:
            raise MissingConversionInputs("None of the supplied inputs '%s' "
                                          "can produce output '%s'. It can be "
                                          "produced by the following "
                                          "combinations '%s'." % (
                                                input_schema,
                                                okey, deps.keys()))

    # Output types must be all "real" or all "complex"
    if not all(dtypes[0] == dt for dt in dtypes[1:]):
        raise ValueError("Output data types differ %s" % dtypes)

    return mapping, input_shape, output_shape, dtypes[0]


def schema_to_tuple(schema):
    if isinstance(schema, (tuple, list)):
        return tuple(schema_to_tuple(s) for s in schema)
    else:
        return schema


def _key_fn(inputs, input_schema, output_schema):
    return (inputs.dtype,
            schema_to_tuple(input_schema),
            schema_to_tuple(output_schema))


_TEMPLATE_PATH = pjoin("model", "coherency", "cuda", "conversion.cu.j2")


@memoize_on_key(_key_fn)
def _generate_kernel(inputs, input_schema, output_schema):
    mapping, in_shape, out_shape, out_dtype = stokes_convert_setup(
                                                inputs,
                                                input_schema,
                                                output_schema)

    # Flatten input and output shapes
    # Check that number elements are the same
    in_elems = reduce(mul, in_shape, 1)
    out_elems = reduce(mul, out_shape, 1)

    if in_elems != out_elems:
        raise ValueError("Number of input_schema elements %s "
                         "and output schema elements %s "
                         "must match for CUDA kernel." %
                         (in_shape, out_shape))

    # Infer the output data type
    if out_dtype == "real":
        if np.iscomplexobj(inputs):
            out_dtype = inputs.real.dtype
        else:
            out_dtype = inputs.dtype
    elif out_dtype == "complex":
        if np.iscomplexobj(inputs):
            out_dtype = inputs.dtype
        else:
            out_dtype = np.result_type(inputs.dtype, np.complex64)
    else:
        raise ValueError("Invalid setup dtype %s" % out_dtype)

    cuda_out_dtype = cuda_type(out_dtype)
    assign_exprs = []

    # Render the assignment expression for each element
    for (c1, c1i), (c2, c2i), outi, template_fn in mapping:
        # Flattened indices
        flat_outi = np.ravel_multi_index(outi, out_shape)
        render = jinja_env.from_string(template_fn).render
        kwargs = {c1: "in[%d]" % np.ravel_multi_index(c1i, in_shape),
                  c2: "in[%d]" % np.ravel_multi_index(c2i, in_shape),
                  "out_type": cuda_out_dtype}

        expr_str = render(**kwargs)
        assign_exprs.append("out[%d] = %s;" % (flat_outi, expr_str))

    # Now render the main template
    render = jinja_env.get_template(_TEMPLATE_PATH).render
    name = "stokes_convert"
    code = render(kernel_name=name,
                  input_type=cuda_type(inputs.dtype),
                  output_type=cuda_type(out_dtype),
                  assign_exprs=assign_exprs,
                  elements=in_elems)

    # cuda block, flatten non-schema dims into a single source dim
    blockdimx = 512
    block = (blockdimx, 1, 1)

    return (cp.RawKernel(code, name), block, in_shape, out_shape, out_dtype)


@requires_optional('cupy', opt_import_error)
def convert(inputs, input_schema, output_schema):
    (kernel, block,
     in_shape, out_shape, dtype) = _generate_kernel(inputs,
                                                    input_schema,
                                                    output_schema)

    # Flatten non-schema input dimensions,
    # from inspection of the cupy reshape code,
    # this incurs a copy when inputs is non-contiguous
    nsrc = reduce(mul, inputs.shape[:-len(in_shape)], 1)
    nelems = reduce(mul, in_shape, 1)

    rinputs = inputs.reshape(nsrc, nelems)
    assert rinputs.flags.c_contiguous
    grid = grids((nsrc, 1, 1), block)

    outputs = cp.empty(shape=rinputs.shape, dtype=dtype)

    try:
        kernel(grid, block, (rinputs, outputs))
    except CompileException:
        log.exception(format_code(kernel.code))
        raise

    shape = inputs.shape[:-len(in_shape)] + out_shape
    outputs = outputs.reshape(shape)
    assert outputs.flags.c_contiguous
    return outputs


try:
    convert.__doc__ = CONVERT_DOCS.substitute(
                                array_type=":class:`cupy.ndarray`")
except AttributeError:
    pass
