# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

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

# stokes_conv = {
#     'XX': {('I', 'Q'): lambda i, q: i + q + 0j},
#     'XY': {('U', 'V'): lambda u, v: u + v*1j},
#     'YX': {('U', 'V'): lambda u, v: u - v*1j},
#     'YY': {('I', 'Q'): lambda i, q: i - q + 0j},
#     'I': {('XX', 'YY'): lambda xx, yy: 0.5*(xx + yy).real,
#           ('RR', 'LL'): lambda rr, ll: 0.5*(rr + ll).real},

#     'Q': {('XX', 'YY'): lambda xx, yy: 0.5*(xx - yy).real,
#           ('RL', 'LR'): lambda rl, lr: 0.5*(rl + lr).real},

#     'U': {('XY', 'YX'): lambda xy, yx: (0.5j*(xy + yx)).real,
#           ('RL', 'LR'): lambda rl, lr: (-0.5j*(rl - lr)).real},

#     'V': {('XY', 'YX'): lambda xy, yx: (-0.5j*(xy - yx)).real,
#           ('RL', 'LR'): lambda rr, ll: (0.5*(rr - ll)).real},
# }


def transformer(inputs, outputs):
    mapping = []

    inputs = np.asarray(inputs)
    outputs = np.asarray(outputs)

    for out_index in np.ndindex(outputs.shape):
        try:
            deps = stokes_deps[outputs[out_index]]
        except KeyError:
            raise ValueError("Unknown output '%s'. "
                             "Known types '%s'"
                             % (deps, STOKES_TYPES))

        found_conv = False

        for (c1, c2), (a, s1, s2) in deps.items():
            try:
                c1 = list(zip(*np.where(inputs == c1)))[0]
            except IndexError:
                continue

            try:
                c2 = list(zip(*np.where(inputs == c2)))[0]
            except IndexError:
                continue

            found_conv = True
            break

        if not found_conv:
            raise ValueError("Unable to covert output '%s'")

        mapping.append((c1, c2, out_index, a, s1, s2))

    mapping = tuple(mapping)

    @numba.jit(nopython=True, nogil=True, cache=True)
    def _numba_xform(input, out_shape):
        output = np.empty(out_shape, dtype=input.dtype)

        for c1, c2, o, a, s1, s2 in mapping:
            i1_idx = (Ellipsis,) + c1
            i2_idx = (Ellipsis,) + c2
            o_idx = (Ellipsis,) + o
            output[o_idx] = a*(s1*input[i1_idx] + s2*input[i2_idx])

        return output

    def _numpy_xform(input):
        outer_input_shape = input.shape[-inputs.ndim:]

        if outer_input_shape != inputs.shape:
            raise ValueError("Expected last dimension(s) "
                             "of input to be '%s' but got '%s'"
                             % (inputs.shape, outer_input_shape))

        return _numba_xform(input, input.shape[:-inputs.ndim] + outputs.shape)

    return _numpy_xform
