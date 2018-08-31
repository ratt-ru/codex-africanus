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

# Correlation dependencies required for reconstructing stokes values
# (corr1, corr2, a, s1, s2). stokes = a*(s1*corr1 + s2*corr2)
CV = namedtuple("Converter", ["a", "s1", "s2"])

stokes_deps = {
    'XX': {('I', 'Q'): CV(1.0 + 0.0j, 1,  1.0 + 0.0j)},
    'XY': {('U', 'V'): CV(1.0 + 0.0j, 1,  0.0 + 1.0j)},
    'YX': {('U', 'V'): CV(1.0 + 0.0j, 1,  0.0 - 1.0j)},
    'YY': {('I', 'Q'): CV(1.0 + 0.0j, 1, -1.0 + 0.0j)},
    'I': {('XX', 'YY'): CV(0.5 + 0.0j, 1,  1),
          ('RR', 'LL'): CV(0.5 + 0.0j, 1,  1)},
    'Q': {('XX', 'YY'): CV(0.5 + 0.0j, 1, -1),
          ('RL', 'LR'): CV(0.5 + 0.0j, 1,  1)},
    'U': {('XY', 'YX'): CV(0.0 + 0.5j, 1,  1),
          ('RL', 'LR'): CV(0.0 - 0.5j, 1, -1)},
    'V': {('XY', 'YX'): CV(0.0 - 0.5j, 1, -1),
          ('RR', 'LL'): CV(0.0 + 0.5j, 1, -1)}
}

stokes_conv = {
    'XX': {('I', 'Q'): lambda i, q: i + q + 0j},
    'XY': {('U', 'V'): lambda u, v: u + v*1j},
    'YX': {('U', 'V'): lambda u, v: u - v*1j},
    'YY': {('I', 'Q'): lambda i, q: i - q + 0j},
    'I': {('XX', 'YY'): lambda xx, yy: 0.5*(xx + yy).real,
          ('RR', 'LL'): lambda rr, ll: 0.5*(rr + ll).real},

    'Q': {('XX', 'YY'): lambda xx, yy: 0.5*(xx - yy).real,
          ('RL', 'LR'): lambda rl, lr: 0.5*(rl + lr).real},

    'U': {('XY', 'YX'): lambda xy, yx: (0.5j*(xy + yx)).real,
          ('RL', 'LR'): lambda rl, lr: (-0.5j*(rl - lr)).real},

    'V': {('XY', 'YX'): lambda xy, yx: (-0.5j*(xy - yx)).real,
          ('RL', 'LR'): lambda rr, ll: (0.5*(rr - ll)).real},
}


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
                raise ValueError("Dimension mismatch %d != %d at depth %d" %
                                 (shape[depth], len(current), depth))

            for i, e in enumerate(current):
                stack.insert(0, (e, current_idx + (i, ), depth + 1))
        else:
            result[current.upper()] = current_idx

    return result, tuple(shape)


def convert(input, input_schema, output_schema):
    input_indices, input_shape = _element_indices_and_shape(input_schema)
    output_indices, output_shape = _element_indices_and_shape(output_schema)

    if input.shape[-len(input_shape):] != input_shape:
        raise ValueError("Last dimension of input doesn't match input schema")

    out_shape = input.shape[:-len(input_shape)] + output_shape
    output = np.empty(out_shape, dtype=input.dtype)

    for okey, out_idx in output_indices.items():
        try:
            deps = stokes_deps[okey]
        except KeyError:
            raise ValueError("Unknown output '%s'. "
                             "Known types '%s'"
                             % (deps, STOKES_TYPES))

        # Find a mapping for which we have inputs
        for (c1, c2), (a, s1, s2) in deps.items():
            # Get indices for both correlations
            try:
                c1_idx = (Ellipsis,) + input_indices[c1]
            except KeyError:
                continue

            try:
                c2_idx = (Ellipsis,) + input_indices[c2]
            except KeyError:
                continue

            out_idx = (Ellipsis,) + out_idx
            output[out_idx] = a*(s1*input[c1_idx] + s2*input[c2_idx])

    return output
