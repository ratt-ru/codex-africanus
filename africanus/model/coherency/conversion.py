# -*- coding: utf-8 -*-


from collections import OrderedDict, deque
from pprint import pformat
from textwrap import fill

import numpy as np

from africanus.util.casa_types import (STOKES_TYPES,
                                       STOKES_ID_MAP)
from africanus.util.docs import DocstringTemplate

stokes_conv = {
    'RR': {('I', 'V'): lambda i, v: i + v + 0j},
    'RL': {('Q', 'U'): lambda q, u: q + u*1j},
    'LR': {('Q', 'U'): lambda q, u: q - u*1j},
    'LL': {('I', 'V'): lambda i, v: i - v + 0j},

    'XX': {('I', 'Q'): lambda i, q: i + q + 0j},
    'XY': {('U', 'V'): lambda u, v: u + v*1j},
    'YX': {('U', 'V'): lambda u, v: u - v*1j},
    'YY': {('I', 'Q'): lambda i, q: i - q + 0j},

    'I': {('XX', 'YY'): lambda xx, yy: (xx + yy).real / 2,
          ('RR', 'LL'): lambda rr, ll: (rr + ll).real / 2},

    'Q': {('XX', 'YY'): lambda xx, yy: (xx - yy).real / 2,
          ('RL', 'LR'): lambda rl, lr: (rl + lr).real / 2},

    'U': {('XY', 'YX'): lambda xy, yx: (xy + yx).real / 2,
          ('RL', 'LR'): lambda rl, lr: (rl - lr).imag / 2},

    'V': {('XY', 'YX'): lambda xy, yx: (xy - yx).imag / 2,
          ('RR', 'LL'): lambda rr, ll: (rr - ll).real / 2},
}


class DimensionMismatch(Exception):
    pass


class MissingConversionInputs(Exception):
    pass


def _element_indices_and_shape(data):
    if not isinstance(data, (tuple, list)):
        data = [data]

    # Shape of the data
    shape = []

    # Each stack element is (list, index, depth)
    queue = deque([(data, (), 0)])
    result = OrderedDict()

    while len(queue) > 0:
        current, current_idx, depth = queue.popleft()

        # First do shape inference
        if len(shape) <= depth:
            shape.append(len(current))
        elif shape[depth] != len(current):
            raise DimensionMismatch("Dimension mismatch %d != %d at depth %d"
                                    % (shape[depth], len(current), depth))

        # Handle each sequence element
        for i, e in enumerate(current):
            # Found a list, recurse
            if isinstance(e, (tuple, list)):
                queue.append((e, current_idx + (i, ), depth + 1))
            # String
            elif isinstance(e, str):
                if e in result:
                    raise ValueError("'%s' defined multiple times" % e)

                result[e] = current_idx + (i, )
            # We have a CASA integer Stokes ID, convert to string
            elif np.issubdtype(type(e), np.integer):
                try:
                    e = STOKES_ID_MAP[e]
                except KeyError:
                    raise ValueError("Invalid id '%d'. "
                                     "Valid id's '%s'"
                                     % (e, pformat(STOKES_ID_MAP)))

                if e in result:
                    raise ValueError("'%s' defined multiple times" % e)

                result[e] = current_idx + (i, )
            else:
                raise TypeError("Invalid type '%s' for element '%s'"
                                % (type(e), e))

    return result, tuple(shape)


def convert_setup(input, input_schema, output_schema):
    input_indices, input_shape = _element_indices_and_shape(input_schema)
    output_indices, output_shape = _element_indices_and_shape(output_schema)

    if input.shape[-len(input_shape):] != input_shape:
        raise ValueError("Last dimension of input doesn't match input schema")

    mapping = []
    dummy = input.dtype.type(0)

    # Figure out how to produce an output from available inputs
    for okey, out_idx in output_indices.items():
        try:
            deps = stokes_conv[okey]
        except KeyError:
            raise ValueError("Unknown output '%s'. Known types '%s'"
                             % (deps, STOKES_TYPES))

        found_conv = False

        # Find a mapping for which we have inputs
        for (c1, c2), fn in deps.items():
            # Get indices for both correlations
            try:
                c1_idx = (Ellipsis,) + input_indices[c1]
            except KeyError:
                continue

            try:
                c2_idx = (Ellipsis,) + input_indices[c2]
            except KeyError:
                continue

            found_conv = True
            out_idx = (Ellipsis,) + out_idx
            # Figure out the data type for this output
            dtype = fn(dummy, dummy).dtype
            mapping.append((c1_idx, c2_idx, out_idx, fn, dtype))
            break

        # We must find a conversion
        if not found_conv:
            raise MissingConversionInputs("None of the supplied inputs '%s' "
                                          "can produce output '%s'. It can be "
                                          "produced by the following "
                                          "combinations '%s'." % (
                                                input_schema,
                                                okey, deps.keys()))

    out_dtype = np.result_type(*[dt for _, _, _, _, dt in mapping])

    return mapping, input_shape, output_shape, out_dtype


def convert_impl(input, mapping, in_shape, out_shape, dtype):
    # Make the output array
    out_shape = input.shape[:-len(in_shape)] + out_shape
    output = np.empty(out_shape, dtype=dtype)

    for c1_idx, c2_idx, out_idx, fn, _ in mapping:
        output[out_idx] = fn(input[c1_idx], input[c2_idx])

    return output


def convert(input, input_schema, output_schema):
    """ See STOKES_DOCS below """

    # Do the conversion
    mapping, in_shape, out_shape, dtype = convert_setup(input,
                                                        input_schema,
                                                        output_schema)

    return convert_impl(input, mapping, in_shape, out_shape, dtype)


CONVERT_DOCS = """
This function converts forward and backward
from stokes ``I,Q,U,V`` to both linear ``XX,XY,YX,YY``
and circular ``RR, RL, LR, LL`` correlations.

For example, we can convert from stokes parameters
to linear correlations:

.. code-block:: python

    stokes.shape == (10, 4, 4)
    corrs = convert(stokes, ["I", "Q", "U", "V"],
                    [['XX', 'XY'], ['YX', 'YY'])

    assert corrs.shape == (10, 4, 2, 2)

Or circular correlations to stokes:

.. code-block:: python

    vis.shape == (10, 4, 2, 2)

    stokes = convert(vis, [['RR', 'RL'], ['LR', 'LL']],
                            ['I', 'Q', 'U', 'V'])

    assert stokes.shape == (10, 4, 4)

``input`` can ``output`` can be arbitrarily nested or ordered lists,
but the appropriate inputs must be present to produce the requested
outputs.

The elements of ``input`` and ``output`` may be strings or integers
representing stokes parameters or correlations. See the Notes
for a full list.


Notes
-----

Only stokes parameters, linear and circular correlations are
currently handled, but the full list of id's and strings as defined
in the `CASA documentation
<https://casacore.github.io/casacore/classcasacore_1_1Stokes.html>`_
is:

.. code-block:: python

    {stokes_type_map}

Parameters
----------
input : $(array_type)
    Complex or floating point input data of shape
    :code:`(dim_1, ..., dim_n, icorr_1, ..., icorr_m)`
input_schema : list of str or int
    A schema describing the :code:`icorr_1, ..., icorr_m`
    dimension of ``input``. Must have the same shape as
    the last dimensions of ``input``.
output_schema : list of str or int
    A schema describing the :code:`ocorr_1, ..., ocorr_n`
    dimension of the return value.

Returns
-------
 result : $(array_type)
    Result of shape :code:`(dim_1, ..., dim_n, ocorr_1, ..., ocorr_m)`
    The type may be floating point or promoted to complex
    depending on the combinations in ``output``.
"""

# Fill in the STOKES TYPES
_map_str = ", ".join(["%s: %d" % (t, i) for i, t in enumerate(STOKES_TYPES)])
_map_str = "{{ " + _map_str + " }}"
# Indent must match docstrings
_map_str = fill(_map_str, initial_indent='', subsequent_indent=' '*8)
CONVERT_DOCS = DocstringTemplate(CONVERT_DOCS.format(stokes_type_map=_map_str))
del _map_str

try:
    convert.__doc__ = CONVERT_DOCS.substitute(
                                  array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass
