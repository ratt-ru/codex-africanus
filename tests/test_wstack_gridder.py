from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from africanus.coordinates import radec_to_lmn
from africanus.gridding.wstack import w_stacking_layers


def test_w_stacking_layers():
    radec = np.asarray([[-np.pi / 4, np.pi /4]])
    lmn = radec_to_lmn(radec, [0, 0])

    print(radec, lmn)

    layers = w_stacking_layers(10, 100, lmn[:, 0], lmn[:, 1])
