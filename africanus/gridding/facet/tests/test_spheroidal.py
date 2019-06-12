# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from numpy.testing import assert_array_almost_equal
import pytest

from africanus.gridding.facet.spheroidal import np_spheroidal_2d, spheroidal_2d



@pytest.mark.parametrize("support", [10, 11, 12])
def test_spheroidal(support):
    assert_array_almost_equal(np_spheroidal_2d(support),
                              spheroidal_2d(support))
