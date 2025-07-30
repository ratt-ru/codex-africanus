# -*- coding: utf-8 -*-


from collections import OrderedDict, deque
from dataclasses import dataclass
from enum import Enum
import heapq
from pprint import pformat
from textwrap import fill
from typing import Any, Callable, Tuple

import numpy as np
import numpy.typing as npt

from africanus.util.casa_types import STOKES_TYPES, STOKES_ID_MAP
from africanus.util.docs import DocstringTemplate

# Definitions for conversion from stokes to correlations
STOKES_TO_CORR_CONV = {
    "RR": {("I", "V"): lambda i, v: i + v + 0j},
    "RL": {("Q", "U"): lambda q, u: q + u * 1j},
    "LR": {("Q", "U"): lambda q, u: q - u * 1j},
    "LL": {("I", "V"): lambda i, v: i - v + 0j},
    "XX": {("I", "Q"): lambda i, q: i + q + 0j},
    "XY": {("U", "V"): lambda u, v: u + v * 1j},
    "YX": {("U", "V"): lambda u, v: u - v * 1j},
    "YY": {("I", "Q"): lambda i, q: i - q + 0j},
}

# Definitions for conversion from correlations to stokes
CORR_TO_STOKES_CONV = {
    "I": {
        ("XX", "YY"): lambda xx, yy: (xx + yy) / 2,
        ("RR", "LL"): lambda rr, ll: (rr + ll) / 2,
    },
    "Q": {
        ("XX", "YY"): lambda xx, yy: (xx - yy) / 2,
        ("RL", "LR"): lambda rl, lr: (rl + lr) / 2,
    },
    "U": {
        ("XY", "YX"): lambda xy, yx: (xy + yx) / 2,
        ("RL", "LR"): lambda rl, lr: (rl - lr) / 2j,
    },
    "V": {
        ("XY", "YX"): lambda xy, yx: (xy - yx) / 2j,
        ("RR", "LL"): lambda rr, ll: (rr - ll) / 2,
    },
}


CONVERSION_SCHEMA = {**STOKES_TO_CORR_CONV, **CORR_TO_STOKES_CONV}

DataSource = Enum("DataSource", ["Index", "Default"])


@dataclass(slots=True)
class ProductMapping:
    """Defines a mapping between two datasources and a destination via a callable function

    Implements partial ordering such that mapping from actual data (DataSource.Index)
    are preferredto DataSources producing default values (DataSource.Default)
    """

    source_one: Tuple[DataSource, Any]
    source_two: Tuple[DataSource, Any]
    dest_index: Tuple[Any, ...]
    fn: Callable
    dtype: npt.DTypeLike

    @property
    def priority(self):
        """Prefer Index Data sources to Default Data sources"""
        return int(self.source_one[0] == DataSource.Index) + int(
            self.source_two[0] == DataSource.Index
        )

    def __lt__(self, other):
        if not isinstance(other, ProductMapping):
            raise NotImplementedError
        # The priority reversal is intentional
        return self.priority > other.priority


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
            raise DimensionMismatch(
                f"Dimension mismatch {shape[depth]} != {len(current)} at depth {depth}"
            )

        # Handle each sequence element
        for i, e in enumerate(current):
            # Found a list, recurse
            if isinstance(e, (tuple, list)):
                queue.append((e, current_idx + (i,), depth + 1))
            # String
            elif isinstance(e, str):
                if e in result:
                    raise ValueError(f"'{e}' defined multiple times")

                result[e] = current_idx + (i,)
            # We have a CASA integer Stokes ID, convert to string
            elif np.issubdtype(type(e), np.integer):
                try:
                    e = STOKES_ID_MAP[e]
                except KeyError:
                    raise ValueError(
                        f"Invalid id '{e}'. Valid id's '{pformat(STOKES_ID_MAP)}'"
                    )

                if e in result:
                    raise ValueError(f"'{e}' defined multiple times")

                result[e] = current_idx + (i,)
            else:
                raise TypeError(f"Invalid type '{type(e)}' for element '{e}'")

    return result, tuple(shape)


def convert_setup(input, input_schema, output_schema, implicit_stokes):
    input_indices, input_shape = _element_indices_and_shape(input_schema)
    output_indices, output_shape = _element_indices_and_shape(output_schema)

    if input.shape[-len(input_shape) :] != input_shape:
        raise ValueError("Last dimension of input doesn't match input schema")

    mapping = []
    dummy = input.dtype.type(0)

    # Figure out how to produce an output from available inputs
    for okey, out_idx in output_indices.items():
        try:
            deps = CONVERSION_SCHEMA[okey]
        except KeyError:
            raise ValueError(
                f"Unknown output {okey}. Known outputs: {list(CONVERSION_SCHEMA.keys())}"
            )

        # We can substitute defaults for stokes values when converting to correlations
        # This makes it possible to compute mappings such as
        # ['I'] -> ['XX', 'XY', 'YX', 'YY']
        can_substitute_defaults = implicit_stokes and okey in STOKES_TO_CORR_CONV
        okey_mappings = []

        # Find a mapping for which we have inputs or defaults
        for (c1, c2), fn in deps.items():
            if c1 in input_indices:
                c1_src = (DataSource.Index, (Ellipsis,) + input_indices[c1])
            elif can_substitute_defaults:
                c1_src = (DataSource.Default, 0)
            else:
                continue

            if c2 in input_indices:
                c2_src = (DataSource.Index, (Ellipsis,) + input_indices[c2])
            elif can_substitute_defaults:
                c2_src = (DataSource.Default, 0)
            else:
                continue

            out_idx = (Ellipsis,) + out_idx
            # Figure out the data type for this output
            dtype = fn(dummy, dummy).dtype
            heapq.heappush(
                okey_mappings, ProductMapping(c1_src, c2_src, out_idx, fn, dtype)
            )

        if len(okey_mappings) == 0:
            raise MissingConversionInputs(
                f"None of the supplied inputs '{input_schema}' "
                f"can produce output '{okey}'. It can be "
                f"produced by the following "
                f"combinations '{deps.keys()}'."
            )

        # Use the highest priority mapping
        mapping.append(okey_mappings[0])

    out_dtype = np.result_type(*[m.dtype for m in mapping])
    return mapping, input_shape, output_shape, out_dtype


def convert_impl(input, mapping, in_shape, out_shape, dtype):
    # Make the output array
    out_shape = input.shape[: -len(in_shape)] + out_shape
    output = np.empty(out_shape, dtype=dtype)

    for m in mapping:
        c1_type, c1_val = m.source_one
        c2_type, c2_val = m.source_two
        c1_arg = c1_val if c1_type is DataSource.Default else input[c1_val]
        c2_arg = c2_val if c2_type is DataSource.Default else input[c2_val]
        output[m.dest_index] = m.fn(c1_arg, c2_arg)

    return output


def convert(input, input_schema, output_schema, implicit_stokes=False):
    """See STOKES_DOCS below"""

    # Do the conversion
    mapping, in_shape, out_shape, dtype = convert_setup(
        input, input_schema, output_schema, implicit_stokes
    )

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
implicit_stokes : bool
    A flag controlling whether implicitly assuming zeros
    for missing Stokes inputs is allowed. Defaults to False.

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
_map_str = fill(_map_str, initial_indent="", subsequent_indent=" " * 8)
CONVERT_DOCS = DocstringTemplate(CONVERT_DOCS.format(stokes_type_map=_map_str))
del _map_str

try:
    convert.__doc__ = CONVERT_DOCS.substitute(array_type=":class:`numpy.ndarray`")
except AttributeError:
    pass
