# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from collections import namedtuple

import numpy as np

from africanus.constants import c as lightspeed, arcseconds_to_radians
from africanus.util.numba import generated_jit, njit, is_numba_type_none


Smear = namedtuple("Smear", [
    "duvw_dtime",
    "interval",
    "chan_width",
    "time_smear",
    "freq_smear"])


_ARCSEC2RAD = np.deg2rad(1.0/(60*60))


def grid_out_factory(have_sizes, have_grid):
    if have_sizes and have_grid:
        def impl(ny, nx, ncorr, grid, dtype):
            if grid.shape != (ny, nx, ncorr):
                raise ValueError("grid.shape != (ny, nx, ncorr")

            return grid
    elif have_sizes and not have_grid:
        def impl(ny, nx, ncorr, grid, dtype):
            return np.zeros((ny, nx, ncorr), dtype=dtype)
    elif not have_sizes and have_grid:
        def impl(ny, nx, ncorr, grid, dtype):
            return grid
    elif not have_sizes and not have_grid:
        raise ValueError("Either 'nx AND ny' OR 'grid' must be supplied")

    return njit(nogil=True, cache=True)(impl)


@generated_jit(nopython=True, nogil=True, cache=True)
def grid(vis, uvw, flags, weights, freqs,
         w_planes, w_planes_conj, meta,
         grid=None, ny=None, nx=None):

    have_ny = not is_numba_type_none(ny)
    have_nx = not is_numba_type_none(nx)
    have_grid = not is_numba_type_none(grid)

    create_grid = grid_out_factory(have_nx and have_ny, have_grid)
    out_dtype = vis.dtype

    def impl(vis, uvw, flags, weights, freqs,
             w_planes, w_planes_conj, meta,
             grid=None, ny=None, nx=None):
        nrow, nchan, ncorr = vis.shape
        grid = create_grid(ny, nx, ncorr, grid, out_dtype)
        ny, nx = grid.shape[:2]

        maxw = meta.maxw
        ref_wave = meta.ref_wave
        overs = meta.oversampling
        nw = len(w_planes)

        if nw != len(w_planes_conj):
            raise ValueError("Number of kernels and conjugates differ")

        for cf, cf_conj in zip(w_planes, w_planes_conj):
            if cf.shape[0] % 2 != 1 or cf.shape[1] % 2 != 1:
                raise ValueError("w-projection kernels must be odd")

            if cf_conj.shape[0] % 2 != 1 or cf_conj.shape[1] % 2 != 1:
                raise ValueError("conjugate w-projection kernels must be odd")

        if ny % 2 != 1 or nx % 2 != 1:
            raise ValueError("Grid must be odd")

        half_y = np.float64(ny // 2)
        half_x = np.float64(nx // 2)

        # # UV scaling in radians
        u_scale = np.float64(arcseconds_to_radians * meta.cell_size_x * nx)
        v_scale = np.float64(arcseconds_to_radians * meta.cell_size_y * ny)

        for r in range(nrow):
            # Extract UVW coordinate, incorporating 1st order polynomial
            w = uvw[r, 2]
            u = uvw[r, 0] + w*meta.cu
            v = uvw[r, 1] + w*meta.cv

            for f in range(nchan):
                # Exit early if all correlations flagged
                if np.all(flags[r, f, :]):
                    continue

                inv_wave = freqs[f] / lightspeed
                # Look up the w projection kernel
                wi = int(np.round((nw - 1)*np.abs(w)*ref_wave*inv_wave/maxw))

                if wi >= nw:
                    continue

                w_kernel = w_planes[wi] if w > 0 else w_planes_conj[wi]
                ncy, ncx = w_kernel.shape

                # This calculation probably depends on an odd support size
                support_y = ncy // overs
                support_x = ncx // overs
                half_sup_y = support_y // 2
                half_sup_x = support_x // 2

                # Scale UV coordinate into the [0:ny, 0:nx]
                # coordinate system
                pos_x = u * u_scale * inv_wave + half_x
                pos_y = v * v_scale * inv_wave + half_y

                # Snap to grid point (float and integer)
                loc_yf = np.round(pos_y)
                loc_xf = np.round(pos_x)
                loc_yi = int(loc_yf)
                loc_xi = int(loc_xf)

                # Only degrid vis if the full support lies within the grid
                if (loc_xi - half_sup_x < 0 or
                    loc_xi + half_sup_x >= nx or
                    loc_yi - half_sup_y < 0 or
                        loc_yi + half_sup_y >= ny):
                    continue

                # Oversampling index in a [-n/2, n/2] coordinate system
                overs_y = int(np.round((loc_yf - pos_y)*overs))
                overs_x = int(np.round((loc_xf - pos_x)*overs))

                # Shift to a [0, n] coordinate system
                overs_y = (overs_y + ((3 * overs) // 2)) % overs
                overs_x = (overs_x + ((3 * overs) // 2)) % overs

                # Dereference the appropriate kernel
                # associated with these oversampling factors
                w_kernel_exp = w_kernel.reshape((overs, overs,
                                                 support_y, support_x))

                sub_kernel = w_kernel_exp[overs_y, overs_x, :, :]

                # Iterate over the convolution kernel
                # accumulating values on the grid
                for sy in range(0, support_y):
                    gy = loc_yi + sy  # Grid position in y

                    for sx in range(0, support_x):
                        gx = loc_xi + sx                # Grid position in x
                        conv_weight = sub_kernel[sy, sx]     # Filter value

                        for c in range(ncorr):
                            # Ignore if flagged
                            if flags[r, f, c] != 0:
                                continue

                            # Accumulate visibility onto the grid
                            grid[gy, gx, c] += (vis[r, f, c] *
                                                weights[r, f, c] *
                                                conv_weight)

        return grid

    return impl


def vis_out_factory(have_vis):
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

    create_output = vis_out_factory(not is_numba_type_none(vis))

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

        for cf, cf_conj in zip(w_planes, w_planes_conj):
            if cf.shape[0] % 2 != 1 or cf.shape[1] % 2 != 1:
                raise ValueError("w-projection kernels must be odd")

            if cf_conj.shape[0] % 2 != 1 or cf_conj.shape[1] % 2 != 1:
                raise ValueError("conjugate w-projection kernels must be odd")

        if ny % 2 != 1 or nx % 2 != 1:
            raise ValueError("Grid must be odd")

        half_y = np.float64(ny // 2)
        half_x = np.float64(nx // 2)

        # # UV scaling in radians
        u_scale = np.float64(arcseconds_to_radians * meta.cell_size_x * nx)
        v_scale = np.float64(arcseconds_to_radians * meta.cell_size_y * ny)

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
                ncy, ncx = w_kernel.shape

                # This calculation probably depends on an odd support size
                support_y = ncy // overs
                support_x = ncx // overs
                half_sup_y = support_y // 2
                half_sup_x = support_x // 2

                # Scale UV coordinate into the [0:ny, 0:nx]
                # coordinate system
                pos_x = u * u_scale * inv_wave + half_x
                pos_y = v * v_scale * inv_wave + half_y

                # Snap to grid point (float and integer)
                loc_yf = np.round(pos_y)
                loc_xf = np.round(pos_x)
                loc_yi = int(loc_yf)
                loc_xi = int(loc_xf)

                # Only degrid vis if the full support lies within the grid
                if (loc_xi - half_sup_x < 0 or
                    loc_xi + half_sup_x >= nx or
                    loc_yi - half_sup_y < 0 or
                        loc_yi + half_sup_y >= ny):
                    continue

                # Oversampling index in a [-n/2, n/2] coordinate system
                overs_y = int(np.round((loc_yf - pos_y)*overs))
                overs_x = int(np.round((loc_xf - pos_x)*overs))

                # Shift to a [0, n] coordinate system
                overs_y = (overs_y + ((3 * overs) // 2)) % overs
                overs_x = (overs_x + ((3 * overs) // 2)) % overs

                # Dereference the appropriate kernel
                # associated with these oversampling factors
                w_kernel_exp = w_kernel.reshape((overs, overs,
                                                 support_y, support_x))

                sub_kernel = w_kernel_exp[overs_y, overs_x, :, :]

                # Zero accumulation buffers
                vis_scratch[:] = 0

                # Iterate over the convolution kernel
                # accumulating values from the grid
                for sy in range(0, support_y):
                    gy = loc_yi + sy  # Grid position in y

                    for sx in range(0, support_x):
                        gx = loc_xi + sx                   # Grid position in x
                        conv_weight = sub_kernel[sy, sx]   # Filter value

                        for c in range(ncorr):
                            vis_scratch[c] += grid[gy, gx, c] * conv_weight

                vis[r, f, :] = vis_scratch

        return vis

    return impl
