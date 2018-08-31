# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple, OrderedDict

import numba
import numpy as np

from ..compatibility import zip

STOKES_TYPES = [
    "Undefined",
    "I",
    "Q",
    "U",
    "V",
    "RR",
    "RL",
    "LR",
    "LL",
    "XX",
    "XY",
    "YX",
    "YY",
    "RX",
    "RY",
    "LX",
    "LY",
    "XR",
    "XL",
    "YR",
    "YL",
    "PP",
    "PQ",
    "QP",
    "QQ",
    "RCircular",
    "LCircular",
    "Linear",
    "Ptotal",
    "Plinear",
    "PFtotal",
    "PFlinear",
    "Pangle"]
"""
List of stokes types as defined in
Measurement Set 2.0 as per Stokes.h in casacore:
https://casacore.github.io/casacore/classcasacore_1_1Stokes.html
"""


STOKES_TYPE_MAP = {k: i for i, k in enumerate(STOKES_TYPES)}
"""
Map of stokes type enumerations as defined in
Measurement Set 2.0 as per Stokes.h in casacore:
https://casacore.github.io/casacore/classcasacore_1_1Stokes.html
"""

stokes_conv = {
    'RR': {('I', 'V'): lambda i, v: i + v + 0j},
    'RL': {('Q', 'U'): lambda q, u: q + u*1j},
    'LR': {('Q', 'U'): lambda q, u: q - u*1j},
    'LL': {('I', 'V'): lambda i, v: i - v + 0j},

    'XX': {('I', 'Q'): lambda i, q: i + q + 0j},
    'XY': {('U', 'V'): lambda u, v: u + v*1j},
    'YX': {('U', 'V'): lambda u, v: u - v*1j},
    'YY': {('I', 'Q'): lambda i, q: i - q + 0j},

    'I': {('XX', 'YY'): lambda xx, yy: 0.5*(xx + yy).real,
          ('RR', 'LL'): lambda rr, ll: 0.5*(rr + ll).real},

    'Q': {('XX', 'YY'): lambda xx, yy: 0.5*(xx - yy).real,
          ('RL', 'LR'): lambda rl, lr: 0.5*(rl + lr).real},

    'U': {('XY', 'YX'): lambda xy, yx: (0.5j*(xy + yx)).imag,
          ('RL', 'LR'): lambda rl, lr: (-0.5j*(rl - lr)).real},

    'V': {('XY', 'YX'): lambda xy, yx: (-0.5j*(xy - yx)).real,
          ('RR', 'LL'): lambda rr, ll: (0.5*(rr - ll)).real},
}


class DimensionMismatch(Exception):
    pass


class MissingConversionInputs(Exception):
    pass


def _element_indices_and_shape(data):
    if not isinstance(data, (tuple, list)):
        raise ValueError("data must be a tuple/list")

    stack = [(data, (), 0)]
    result = OrderedDict()
    shape = []

    while len(stack) > 0:
        current, current_idx, depth = stack.pop()

        if isinstance(current, (tuple, list)):
            if len(shape) <= depth:
                shape.append(len(current))
            elif shape[depth] != len(current):
                raise DimensionMismatch("Dimension mismatch %d != %d "
                                        "at depth %d" %
                                        (shape[depth], len(current), depth))

            for i, e in enumerate(current):
                stack.insert(0, (e, current_idx + (i, ), depth + 1))
        else:
            result[current.upper()] = current_idx

    return result, tuple(shape)


def stokes_convert(input, input_schema, output_schema):
    """
    This function converts forward and backward
    from stokes ``I,Q,U,V`` to both linear ``XX,XY,YX,YY``
    and circular ``RR, RL, LR, LL`` correlations.

    For example, we can convert from stokes parameters
    to linear correlations:

    .. code-block:: python

        stokes.shape == (10, 4, 4)
        vis = stokes_convert(stokes, ["I", "Q", "U", "V"],
                             [['XX', 'XY'], ['YX', 'YY'])

        assert vis.shape == (10, 4, 2, 2)

    Or circular correlations to stokes:

    .. code-block:: python

        vis.shape == (10, 4, 2, 2)

        stokes = stokes_convert(vis, [['RR', 'RL'], ['LR', 'LL']],
                                ['I', 'Q', 'U'', 'V'])

        assert stokes.shape == (10, 4, 4)

    ``input`` can ``output`` can be arbitrarily nested or ordered lists,
    but the appropriate inputs must be present to produce the requested
    outputs.

    Parameters
    ----------
    input : :class:`numpy.ndarray`
        Complex or floating point input data of shape
        :code:`(dim_1, ..., dim_n, icorr_1, ..., icorr_m)`
    input_schema : list or list of lists
        A schema describing the :code:`icorr_1, ..., icorr_m`
        dimension of ``input``. Must have the same shape as
        the last dimensions of ``input``.
    output_schema : list or list of lists
        A schema describing the :code:`ocorr_1, ..., ocorr_n`
        dimension of the return value.

    Returns
    -------
    :class:`numpy.ndarray`
        Result of shape :code:`(dim_1, ..., dim_n, ocorr_1, ..., ocorr_m)`
    """

    input_indices, input_shape = _element_indices_and_shape(input_schema)
    output_indices, output_shape = _element_indices_and_shape(output_schema)

    if input.shape[-len(input_shape):] != input_shape:
        raise ValueError("Last dimension of input doesn't match input schema")

    mapping = []
    dummy = input.flat[0]

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

    # Make the output array
    out_shape = input.shape[:-len(input_shape)] + output_shape
    out_dtype = np.result_type(*[dt for _, _, _, _, dt in mapping])
    output = np.empty(out_shape, dtype=out_dtype)

    # Do the conversion
    for c1_idx, c2_idx, out_idx, fn, _ in mapping:
        output[out_idx] = fn(input[c1_idx], input[c2_idx])

    return output
