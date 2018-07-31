from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from africanus.coordinates import radec_to_lmn, lmn_to_radec


def test_radec_to_lmn():
    radec = np.random.random((10, 2))*np.pi
    phase_centre = np.random.random(2)*np.pi

    lmn = radec_to_lmn(radec, phase_centre)
    result = lmn_to_radec(lmn, phase_centre)
    result = (result + np.pi) % (2 * np.pi) - np.pi

    print(radec)
    print(result)
    print(np.abs(radec - result))

    assert np.allclose(result, radec)
