# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np


_ARCSEC2RAD = np.deg2rad(1.0/(60*60))


def grid(vis, uvw, ref_wave, conv_filter, oversample,
         cell_size, grid):

    nx = grid.shape[0]
    half_x = nx // 2
    half_support = (conv_filter.shape[0] // oversample) // 2

    # Similarity Theorem
    # https://www.cv.nrao.edu/course/astr534/FTSimilarity.html
    # Scale UV coordinates
    # Note u => x and v => y
    u_scale = _ARCSEC2RAD * cell_size * nx

    for r in range(vis.shape[0]):
        scaled_u = uvw[r, 0] * u_scale

        for f in range(vis.shape[1]):
            exact_u = half_x + (scaled_u / ref_wave[f])
            disc_u = int(np.round(exact_u))

            frac_u = exact_u - disc_u
            base_os_u = int(np.round(frac_u*oversample))
            base_os_u = (base_os_u + ((3 * oversample) // 2)) % oversample

            lower_u = disc_u - half_support      # Inclusive
            upper_u = disc_u + half_support + 1  # Exclusive

            print(exact_u, disc_u, lower_u, upper_u)

            for ui, grid_u in enumerate(range(lower_u, upper_u)):
                conv_weight = conv_filter[base_os_u + ui*oversample]

                grid[grid_u] += vis[r, f] * conv_weight
