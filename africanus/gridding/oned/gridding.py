# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np


_ARCSEC2RAD = np.deg2rad(1.0/(60*60))


nearest_neighbour = False


def grid(vis, uvw, ref_wave, conv_filter, oversample,
         cell_size, grid):

    nx = grid.shape[0]

    assert nx % 2 == 1
    assert conv_filter.shape[0] % 2 == 1

    half_support = (conv_filter.shape[0] // oversample) // 2
    centre_x = nx // 2

    # Similarity Theorem
    # https://www.cv.nrao.edu/course/astr534/FTSimilarity.html
    # Scale UV coordinates
    # Note u => x and v => y
    u_scale = _ARCSEC2RAD * cell_size * nx

    for r in range(vis.shape[0]):
        scaled_u = uvw[r, 0] * u_scale

        for f in range(vis.shape[1]):
            exact_u = centre_x + (scaled_u / ref_wave[f])
            disc_u = int(np.round(exact_u))

            frac_u = exact_u - disc_u
            base_os_u = int(np.round(frac_u*oversample))
            saved = base_os_u
            base_os_u = (base_os_u + ((3 * oversample) // 2)) % oversample

            lower_u = disc_u - half_support               # Inclusive
            upper_u = disc_u + half_support + 1           # Exclusive

            print("\n")
            print("exact %.3f discrete %.3f frac_u*os %.3f frac %.3f "
                  "os_u %.3f lower_u %.3f upper_u %.3f "
                  % (exact_u, disc_u, frac_u*oversample,
                     frac_u, base_os_u, lower_u, upper_u))

            print(exact_u, disc_u, lower_u, upper_u)

            if nearest_neighbour is True:
                grid[disc_u] += vis[r, f]
            else:
                for ui, grid_u in enumerate(range(lower_u, upper_u)):
                    conv_weight = conv_filter[base_os_u + ui*oversample]
                    grid[grid_u] += vis[r, f] * conv_weight
