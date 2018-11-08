# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from africanus.constants import c
from africanus.filters.kaiser_bessel_filter import kaiser_bessel_with_sinc
from africanus.gridding.simple.gridding import _ARCSEC2RAD


@pytest.mark.parametrize("exact_u", [np.array([8.5])])
@pytest.mark.parametrize("full_support, oversample, beta", [[7, 7, 2.4]])
@pytest.mark.parametrize("ref_wave", [c / (0.5*(.856e9 + 2*.856e9))])
@pytest.mark.parametrize("nx", [16])
@pytest.mark.parametrize("cell_size", [2.0])
@pytest.mark.parametrize("plot", [False])
def test_oversampling(exact_u, full_support,
                      oversample, beta,
                      ref_wave, nx,
                      cell_size, plot):
    # The following illustrates the indices of a filter
    # with a support of 3 and oversampling by a factor of 5.
    # | indicates filter index whereas + indicates oversampling index
    #
    #  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14
    #  +   +   |   +   +   +   +   |   +   +   +   +   |   +   +

    # In the example above 3 x 5 = 15
    W = full_support*oversample
    # In the example above, 5 // 2, or the * points
    # on either side of the end |
    base_os = oversample // 2 if oversample % 2 == 1 else (oversample // 2) - 1

    half_support = full_support // 2
    half_x = nx // 2

    exact_u += half_x
    disc_u = np.round(exact_u).astype(np.int32)

    if np.any(disc_u < 0) or np.any(disc_u > nx):
        raise ValueError("Pick values inside the grid [%s-%s]\n%s"
                         % (0, nx, disc_u))

    conv_filter = kaiser_bessel_with_sinc(full_support, oversample, beta=beta)

    # Find oversampling index by multiplying the fractional difference
    # by the oversampling factor. Because of the np.round,
    # the index runs from [-0.5, 0.5]*oversample
    frac_u = exact_u - disc_u
    base_os_u = np.round(frac_u*oversample).astype(np.int32)
    # As per wsclean
    orig_os_u = base_os_u
    # This changes the index from
    # [-2, -1,  0,  1,  2, -2, -1,  0   1,  2, -2, -1,  0,  1,  2]    to
    # [ 0,  1,  2,  3,  4,  0,  1,  2,  3,  4,  0,  1,  2,  3,  4]
    base_os_u = (base_os_u + ((3*oversample) // 2)) % oversample

    print("u: %.3f disc_u: %d frac_u: %.3f orig_os_u: %d "
          "base_os: %d base_os_u: %d frac_u*os %.3f"
          % (exact_u, disc_u, frac_u, orig_os_u,
             base_os, base_os_u, frac_u*oversample))

    if plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.fail("plotting requested but could not import matplotlib")

        x = np.arange(0, nx)
        f, axes = plt.subplots(2, 1)
        f.set_figwidth(80)
        f.set_figheight(60)
        axes[0].scatter(exact_u, np.ones_like(exact_u)*0.25, color='blue')
        axes[0].scatter(disc_u, np.ones_like(exact_u)*0.1, color='red')
        axes[0].set_xticks(x)
        axes[0].set_ylim(0, 0.5)
        axes[0].set_xlabel("grid position")

        x = np.arange(conv_filter.size)
        filter_points = base_os + np.arange(full_support)*oversample

        axes[1].plot(x, conv_filter)
        axes[1].vlines(filter_points,
                       ymin=conv_filter.min(),
                       ymax=0.1 * conv_filter.max())
        axes[1].scatter(filter_points,
                        np.ones_like(filter_points)*0,
                        color='red')
        axes[1].set_xlabel("filter position")
        axes[1].set_xticks(x)

        oversample_points = base_os_u + np.arange(full_support)*oversample
        axes[1].scatter(oversample_points,
                        np.ones_like(oversample_points)*0,
                        color='blue')

        plt.show()
