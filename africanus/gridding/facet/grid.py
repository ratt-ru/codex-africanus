# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from collections import namedtuple

import numpy as np

from africanus.constants import c as lightspeed
from africanus.util.numba import generated_jit, njit, is_numba_type_none


Metadata = namedtuple("Metadata", [
    # Facet phase centre
    "l0", "m0",
    # Reference wavelength, maximum W coordinate
    "ref_wave", "maxw",
    # Oversampling factor
    "oversampling",
    # Cell Size in X and Y
    "cell_size_x", "cell_size_y",
    # First polynomial coefficients
    "cu", "cv"])


Smear = namedtuple("Smear", [
    "duvw_dtime",
    "interval",
    "chan_width",
    "time_smear",
    "freq_smear"])


def output_factory(have_vis):
    if have_vis:
        def impl(nrow, nchan, ncorr, vis, dtype):
            if vis.shape != (nrow, nchan, ncorr):
                raise ValueError("(nrow, nchan, ncorr) != vis.shape")

            return vis
    else:
        def impl(nrow, nchan, ncorr, vis, dtype):
            return np.zeros((nrow, nchan, ncorr), dtype=dtype)

    return njit(nogil=True, cache=True)(impl)


@generated_jit(nopython=True, nogil=True, cache=True)
def degrid(grid, uvw, freqs,
           w_planes, w_planes_conj, meta,
           vis=None):

    _ARCSEC2RAD = np.deg2rad(1.0/(60*60))

    create_output = output_factory(not is_numba_type_none(vis))

    def impl(grid, uvw, freqs,
             w_planes, w_planes_conj, meta,
             vis=None):
        nrow = uvw.shape[0]
        nchan = freqs.shape[0]
        ny, nx, ncorr = grid.shape
        n0 = np.sqrt(1.0 - meta.l0**2 - meta.m0**2) - 1.0  # noqa
        maxw = meta.maxw
        ref_wave = meta.ref_wave
        overs = meta.oversampling
        nw = len(w_planes)

        vis = create_output(nrow, nchan, ncorr, vis, grid.dtype)

        if nw != len(w_planes_conj):
            raise ValueError("Number of kernels and conjugates differ")

        if ny % 2 != 1 or nx % 2 != 1:
            raise ValueError("Grid must be odd")

        centre_y = ny // 2
        centre_x = nx // 2

        # UV scaling in radians
        u_scale = _ARCSEC2RAD * meta.cell_size_x * nx
        v_scale = _ARCSEC2RAD * meta.cell_size_y * ny

        vis_scratch = np.empty((ncorr,), dtype=vis.dtype)

        for r in range(nrow):
            # Extract UVW coordinate, incorporating 1st order polynomial
            w = uvw[r, 2]
            u = uvw[r, 0] + w*meta.cu
            v = uvw[r, 1] + w*meta.cv

            for f in range(nchan):
                inv_wave = freqs[f] / lightspeed
                # Look up the w projection kernel
                wi = int(np.round((nw - 1)*np.abs(w)*ref_wave*inv_wave/maxw))

                if wi >= nw:
                    continue

                w_kernel = w_planes[wi] if w > 0 else w_planes_conj[wi]

                ncy = w_kernel.shape[0]
                ncx = w_kernel.shape[1]

                # This calculation probably depends on an odd support size
                # support_x = int(((ncx // overs) - 1) // 2)
                # support_y = int(((ncy // overs) - 1) // 2)
                # Full support
                # support_cf = ncx // overs

                support_y = ncy // overs
                support_x = ncx // overs
                half_sup_y = support_y // 2
                half_sup_x = support_x // 2

                w_kernel_exp = w_kernel.reshape((overs, overs,
                                                 ncy // overs, ncx // overs))

                # Scale UV coordinate.
                pos_x = u * u_scale * inv_wave + centre_x
                pos_y = v * v_scale * inv_wave + centre_y

                # Snap to grid
                loc_y = int(np.round(pos_y))
                loc_x = int(np.round(pos_x))

                # Only degrid vis if the full support lies within the grid
                if (loc_x - half_sup_x < 0 or
                    loc_x + half_sup_x >= nx or
                    loc_y - half_sup_y < 0 or
                        loc_y + half_sup_y >= ny):
                    continue

                # Location within the oversampling
                off_y = int(np.round((loc_y - pos_y)*overs))
                off_x = int(np.round((loc_x - pos_x)*overs))

                subgrid = w_kernel_exp[off_y, off_x, :, :]
                vis_scratch[:] = 0.0

                for sy in range(0, support_y):
                    gy = loc_y + sy

                    for sx in range(0, support_x):
                        gx = loc_x + sx
                        weight = subgrid[sy, sx]

                        for c in range(ncorr):
                            vis_scratch[c] += grid[gy, gx, c]*weight

                vis[r, f, :] = vis_scratch

        return vis

    return impl
