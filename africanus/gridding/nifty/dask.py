# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

try:
    import dask.array as da
    import nifty_gridder as ng
except ImportError as e:
    import_error = e
else:
    import_error = None

from africanus.util.requirements import requires_optional


def create_baselines(uvw, chan_freq):
    """ Wrapper function for creating baseline mappings per row chunk """
    chan_freq = np.concatenate(chan_freq)
    return ng.Baselines(uvw[0], chan_freq)


def create_indices(baselines, grid_config, flag,
                   chan_begin, chan_end, wmin, wmax):
    """ Wrapper function for creating indices per row chunk """
    return ng.getIndices(baselines, grid_config, flag[0],
                         chan_begin, chan_end, wmin, wmax)


def grid_data(baselines, grid_config, indices, vis):
    """ Wrapper function for creating a grid of visibilities per row chunk """
    flat_vis = baselines.ms2vis(vis[0], indices)
    grid = ng.vis2grid_c(baselines, grid_config, indices, flat_vis)
    return grid[None, :, :]


def grid_config_reduce(self):
    return (self.__class__, (self.Nxdirty(), self.Nydirty(), self.Epsilon(),
                             self.Pixsize_x(), self.Pixsize_y()))


class GridderConfigWrapper(object):
    def __init__(self, nx=1024, ny=1024, eps=2e-13,
                 cell_size_x=2.0, cell_size_y=2.0):
        self.nx = nx
        self.ny = ny
        self.csx = cell_size_x
        self.csy = cell_size_y
        self.eps = eps
        self.grid_config = ng.GridderConfig(nx, ny, eps,
                                            cell_size_x,
                                            cell_size_y)

    @property
    def object(self):
        return self.grid_config

    def __reduce__(self):
        return (GridderConfigWrapper,
                (self.nx, self.ny, self.eps, self.csx, self.csy))


@requires_optional("dask.array", "nifty_gridder", import_error)
def grid_config(nx=1024, ny=1024, cell_size_x=2.0, cell_size_y=2.0, eps=2e-13):
    return GridderConfigWrapper(nx, ny, eps, cell_size_x, cell_size_y)


@requires_optional("dask.array", "nifty_gridder", import_error)
def grid(vis, uvw, flags, weights, frequencies, grid_config,
         wmin=-1e16, wmax=1e16):
    # Create a baseline object per row chunk
    baselines = da.blockwise(create_baselines, ("row",),
                             uvw, ("row", "uvw"),
                             frequencies, ("chan",),
                             dtype=np.object)

    if len(frequencies.chunks[0]) != 1:
        raise ValueError("Chunking in channel unsupported")

    gc = grid_config.object
    grids = []

    for corr in range(vis.shape[2]):
        corr_flags = flags[:, :, corr]

        indices = da.blockwise(create_indices, ("row",),
                               baselines, ("row",),
                               gc, None,
                               corr_flags, ("row", "chan"),
                               0, None,
                               frequencies.shape[0], None,
                               wmin, None,
                               wmax, None,
                               dtype=np.int32)

        grid = da.blockwise(grid_data, ("row", "nu", "nv"),
                            baselines, ("row",),
                            gc, None,
                            indices, ("row",),
                            vis[:, :, corr], ("row", "chan"),
                            new_axes={"nu": gc.Nu(), "nv": gc.Nv()},
                            adjust_chunks={"row": 1},
                            dtype=np.complex128)

        grid = grid.sum(axis=0)
        grids.append(grid)

    return da.stack(grids, axis=2)


def create_dirty(grid_config, grid):
    """ Wrapper function for creating a dirty image """
    grids = [grid_config.grid2dirty_c(grid[:, :, c])
             for c in range(grid.shape[2])]

    return np.stack(grids, axis=2)


@requires_optional("dask.array", "nifty_gridder", import_error)
def dirty(grid, grid_config):
    gc = grid_config.object
    nx = gc.Nxdirty()
    ny = gc.Nydirty()

    return da.blockwise(create_dirty, ("nx", "ny", "corr"),
                        gc, None,
                        grid, ("nx", "ny", "corr"),
                        adjust_chunks={"nx": nx, "ny": ny},
                        dtype=grid.dtype)
