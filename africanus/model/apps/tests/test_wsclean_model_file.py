# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.testing import assert_array_almost_equal


from africanus.model.apps.wsclean_file_model import wsclean


def test_wsclean_model_file(wsclean_model_file):
    sources = wsclean(wsclean_model_file)

    name, stype, _, _, I, spi, log_si, ref_freq, _, _, _ = sources

    # Seven sources
    assert (len(I) == len(spi) == len(log_si) == len(ref_freq) == 7)
    # Missing reference frequency set in the last
    assert (ref_freq[-1] == ref_freq[0] and
            name[-1] == "s1c2" and
            stype[-1] == "GAUSSIAN")
