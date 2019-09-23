# -*- coding: utf-8 -*-


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
Measurement Set 2.0 and Stokes.h in casacore:
https://casacore.github.io/casacore/classcasacore_1_1Stokes.html
"""


STOKES_TYPE_MAP = {k: i for i, k in enumerate(STOKES_TYPES)}
"""
Map of stokes type to enumeration as defined in
Measurement Set 2.0 and Stokes.h in casacore:
https://casacore.github.io/casacore/classcasacore_1_1Stokes.html
"""

STOKES_ID_MAP = {v: k for k, v in STOKES_TYPE_MAP.items()}
"""
Map of stokes ID to stokes type string as defined in
Measurement Set 2.0 and Stokes.h in casacore:
https://casacore.github.io/casacore/classcasacore_1_1Stokes.html
"""
