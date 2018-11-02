# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest


from africanus.filters.kaiser_bessel_filter import (
        kaiser_bessel_with_sinc,
        wsclean_kaiser_bessel_with_sinc)


@pytest.mark.xfail
@pytest.mark.parametrize("full_support, oversampling, beta", [[21, 27, 2.4]])
@pytest.mark.parametrize("plot", [True])
def test_kaiser_bessel_filter(full_support, oversampling, beta, plot):
    wsclean_filter = wsclean_kaiser_bessel_with_sinc(full_support,
                                                     oversampling,
                                                     beta)

    afr_filter = kaiser_bessel_with_sinc(full_support, oversampling,
                                         beta, normalise=False)

    if plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pass
        else:
            x = np.arange(wsclean_filter.size)
            plt.plot(x, wsclean_filter)
            plt.plot(x, afr_filter)
            plt.show()

    print(np.abs(wsclean_filter - afr_filter).max())

    print(wsclean_filter.dtype, afr_filter.dtype)

    assert np.allclose(wsclean_filter, afr_filter)
