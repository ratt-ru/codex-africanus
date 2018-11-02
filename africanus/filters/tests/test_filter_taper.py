# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from africanus.filters import convolution_filter
from africanus.filters.filter_tapers import taper as filter_taper


@pytest.mark.parametrize("filter_type", ["kaiser-bessel"])
@pytest.mark.parametrize("full_support, oversampling", [[7, 7]])
@pytest.mark.parametrize("ny, nx", [[1024, 1024]])
def test_filter_taper(filter_type, full_support, oversampling, ny, nx):
    conv_filter = convolution_filter(filter_type, full_support, oversampling)
    taper = filter_taper("kaiser-bessel", ny, nx, conv_filter)
