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


def output_factory(have_size, have_grid):
    if have_size and have_grid:
        def impl(ny, nx, grid, dtype):
            if grid.shape[:2] == (ny, nx):
                raise ValueError("(ny, nx) != grid.shape[:2")

            return grid
    elif have_size and not have_grid:
        def impl(ny, nx, grid, dtype):
            return np.zeros((ny, nx), dtype=dtype)
    elif not have_size and have_grid:
        def impl(ny, nx, grid, dtype):
            return grid
    else:
        def impl(ny, nx, grid, dtype):
            raise ValueError("Must supply both ny and nx, or a grid")

    return njit(nogil=True, cache=True)(impl)


@generated_jit(nopython=True, nogil=True, cache=True)
def degrid(vis, uvw, flags, freqs,
           w_planes, w_planes_conj, meta,
           grid=None, ny=None, nx=None):

    _ARCSEC2RAD = np.deg2rad(1.0/(60*60))

    have_grid = not is_numba_type_none(grid)
    have_nx = not is_numba_type_none(nx)
    have_ny = not is_numba_type_none(ny)

    create_output = output_factory(have_ny and have_nx, have_grid)

    def impl(vis, uvw, flags, freqs,
             w_planes, w_planes_conj, meta,
             grid=None, ny=None, nx=None):
        nrow, nchan, ncorr = vis.shape
        n0 = np.sqrt(1.0 - meta.l0**2 - meta.m0**2) - 1.0
        maxw = meta.maxw
        ref_wave = meta.ref_wave
        overs = meta.oversampling
        nw = len(w_planes)

        if nw != len(w_planes_conj):
            raise ValueError("Number of kernels and conjugates differ")

        # Create grid and and check shape
        grid = create_output(ny, nx, grid, vis.dtype)

        if grid.shape[1] % 2 != 1 or grid.shape[0] % 2 != 1:
            raise ValueError("Grid must be odd")

        centre_y = grid.shape[1] // 2
        centre_x = grid.shape[0] // 2
        u_scale = _ARCSEC2RAD * meta.cell_size_x * nx
        v_scale = _ARCSEC2RAD * meta.cell_size_y * ny

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

                ncx = w_kernel.shape[0]
                ncy = w_kernel.shape[1]

                # This calculation probably depends on an odd support size
                support_x = int(((ncx // overs) - 1) // 2)
                support_y = int(((ncy // overs) - 1) // 2)
                # Full support
                support_cf = ncx // overs

                # Scale UV coordinate.
                pos_x = u * u_scale * inv_wave + centre_x
                pos_y = v * v_scale * inv_wave + centre_y

                # Snap to grid
                loc_x = int(np.round(pos_x))
                loc_y = int(np.round(pos_y))

                # Only degrid vis if the full support lies within the grid
                if (loc_x - support_x < 0 or
                    loc_x + support_x >= nx or
                    loc_y - support_y < 0 or
                        loc_y + support_y >= ny):
                    continue

                # Location within the oversampling
                off_x = int(np.round((loc_x - pos_x)*overs))
                off_x += (ncx - 1) // 2
                off_y = int(np.round((loc_y - pos_y)*overs))
                off_y += (ncy - 1) // 2

                io = off_y - support_y*overs
                jo = off_x - support_x*overs
                cfoff = (io * overs + jo) * support_cf**2

                for c in range(ncorr):
                    pass

    return impl
