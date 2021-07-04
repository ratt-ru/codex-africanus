# -*- coding: utf-8 -*-


try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

import numpy as np

try:
    import dask
    import dask.array as da
    from dask.base import normalize_token
    from dask.highlevelgraph import HighLevelGraph
except ImportError as e:
    import_error = e
else:
    import_error = None

try:
    import nifty_gridder as ng
except ImportError:
    nifty_import_err = ImportError("Please manually install nifty_gridder "
                                   "from https://gitlab.mpcdf.mpg.de/ift/"
                                   "nifty_gridder.git")
else:
    nifty_import_err = None

from africanus.util.requirements import requires_optional


class GridderConfigWrapper(object):
    """
    Wraps a nifty GridderConfiguration for pickling purposes.
    """

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


if import_error is None:
    @normalize_token.register(GridderConfigWrapper)
    def normalize_gridder_config_wrapper(gc):
        return normalize_token((gc.nx, gc.ny, gc.csx, gc.csy, gc.eps))


@requires_optional("dask.array", import_error)
@requires_optional("nifty_gridder", nifty_import_err)
def grid_config(nx=1024, ny=1024, eps=2e-13, cell_size_x=2.0, cell_size_y=2.0):
    """
    Returns a wrapper around a NIFTY GridderConfiguration object.


    Parameters
    ----------
    nx : int, optional
        Number of X pixels in the grid. Defaults to 1024.
    ny : int, optional
        Number of Y pixels in the grid. Defaults to 1024.
    cell_size_x : float, optional
        Cell size of the X pixel in arcseconds. Defaults to 2.0.
    cell_size_y : float, optional
        Cell size of the Y pixel in arcseconds. Defaults to 2.0.
    eps : float
        Gridder accuracy error. Defaults to 2e-13


    Returns
    -------
    grid_config : :class:`GridderConfigWrapper`
        The NIFTY Gridder Configuration
    """
    return GridderConfigWrapper(nx, ny, eps, cell_size_x, cell_size_y)


def _nifty_baselines(uvw, chan_freq):
    """ Wrapper function for creating baseline mappings per row chunk """
    assert len(chan_freq) == 1, "Handle multiple channel chunks"
    return ng.Baselines(uvw[0], chan_freq[0])


def _nifty_indices(baselines, grid_config, flag,
                   chan_begin, chan_end, wmin, wmax):
    """ Wrapper function for creating indices per row chunk """
    return ng.getIndices(baselines, grid_config, flag[0],
                         chan_begin, chan_end, wmin, wmax)


def _nifty_grid(baselines, grid_config, indices, vis, weights):
    """ Wrapper function for creating a grid of visibilities per row chunk """
    assert len(vis) == 1 and type(vis) is list
    return ng.ms2grid_c(baselines, grid_config, indices,
                        vis[0], None, weights[0])[None, :, :]


def _nifty_grid_streams(baselines, grid_config, indices,
                        vis, weights, grid_in=None):
    """ Wrapper function for creating a grid of visibilities per row chunk """
    return ng.ms2grid_c(baselines, grid_config, indices,
                        vis, grid_in, weights)


class GridStreamReduction(Mapping):
    """
    tl;dr this is a dictionary that is expanded in place when
    first accessed. Saves memory when pickled for sending
    to the dask scheduler.

    See :class:`dask.blockwise.Blockwise` for further insight.

    Produces graph serially summing coherencies in
    ``stream`` parallel streams.
    """

    def __init__(self, baselines, indices, gc,
                 corr_vis, corr_weights,
                 corr, streams):
        token = dask.base.tokenize(baselines, indices, gc,
                                   corr_vis, corr_weights,
                                   corr, streams)
        self.name = "-".join(("nifty-grid-stream", str(corr), token))
        self.bl_name = baselines.name
        self.idx_name = indices.name
        self.cvis_name = corr_vis.name
        self.wgt_name = corr_weights.name
        self.gc = gc
        self.corr = corr

        self.row_blocks = indices.numblocks[0]
        self.streams = streams

    @property
    def _dict(self):
        if hasattr(self, "_cached_dict"):
            return self._cached_dict
        else:
            self._cached_dict = self._create_dict()
            return self._cached_dict

    def __getitem__(self, key):
        return self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        # Extract dimension blocks
        return self.row_blocks

    def _create_dict(self):
        # Graph dictionary
        layers = {}

        name = self.name
        gc = self.gc
        row_blocks = self.row_blocks
        streams = self.streams
        indices_name = self.idx_name
        baselines_name = self.bl_name
        corr_vis_name = self.cvis_name
        corr_wgt_name = self.wgt_name

        cb = 0  # Assume single channel

        # Split our row blocks by the number of streams
        # For all blocks in a stream, we'll grid those
        # blocks serially, passing one grid into the other
        row_block_chunks = (row_blocks + streams - 1) // streams

        for rb_start in range(0, row_blocks, row_block_chunks):
            rb_end = min(rb_start + row_block_chunks, row_blocks)
            last_key = None

            for rb in range(rb_start, rb_end):
                fn = (_nifty_grid_streams,
                      (baselines_name, rb),
                      gc,
                      (indices_name, rb),
                      (corr_vis_name, rb, cb),
                      (corr_wgt_name, rb, cb),
                      # Re-use grid from last operation if present
                      last_key)

                key = (name, rb, cb)
                layers[key] = fn
                last_key = key

        return layers


class FinalGridReduction(Mapping):
    """
    tl;dr this is a dictionary that is expanded in place when
    first accessed. Saves memory when pickled for sending
    to the dask scheduler.

    See :class:`dask.blockwise.Blockwise` for further insight.

    Produces graph serially summing coherencies in
    ``stream`` parallel streams.
    """

    def __init__(self, grid_stream_reduction):
        self.in_name = grid_stream_reduction.name
        token = dask.base.tokenize(grid_stream_reduction)
        self.name = "grid-stream-reduction-" + token
        self.row_blocks = grid_stream_reduction.row_blocks
        self.streams = grid_stream_reduction.streams

    @property
    def _dict(self):
        if hasattr(self, "_cached_dict"):
            return self._cached_dict
        else:
            self._cached_dict = self._create_dict()
            return self._cached_dict

    def __getitem__(self, key):
        return self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        # Always returns a single block, corresponding to one image
        return 1

    def _create_dict(self):
        # Graph dictionary
        layers = {}

        row_blocks = self.row_blocks
        streams = self.streams
        cb = 0  # Assume single channel

        # Split our row blocks by the number of streams
        # For all blocks in a stream, we'll grid those
        # blocks serially, passing one grid into the other
        row_block_chunks = (row_blocks + streams - 1) // streams
        last_keys = []

        for ob, rb_start in enumerate(range(0, row_blocks, row_block_chunks)):
            rb_end = min(rb_start + row_block_chunks, row_blocks)
            last_keys.append((self.in_name, rb_end - 1, cb))

        key = (self.name, 0, 0)
        task = (sum, last_keys)
        layers[key] = task

        return layers


@requires_optional("dask.array", import_error)
@requires_optional("nifty_gridder", nifty_import_err)
def grid(vis, uvw, flags, weights, frequencies, grid_config,
         wmin=-1e30, wmax=1e30, streams=None):
    """
    Grids the supplied visibilities in parallel. Note that
    a grid is create for each visibility chunk.

    Parameters
    ----------
    vis : :class:`dask.array.Array`
        visibilities of shape :code:`(row, chan, corr)`
    uvw : :class:`dask.array.Array`
        uvw coordinates of shape :code:`(row, 3)`
    flags : :class:`dask.array.Array`
        flags of shape :code:`(row, chan, corr)`
    weights : :class:`dask.array.Array`
        weights of shape :code:`(row, chan, corr)`.
    frequencies : :class:`dask.array.Array`
        frequencies of shape :code:`(chan,)`
    grid_config : :class:`GridderConfigWrapper`
        Gridding Configuration
    wmin : float
        Minimum W coordinate to grid. Defaults to -1e30.
    wmax : float
        Maximum W coordinate to grid. Default to 1e30.
    streams : int, optional
        Number of parallel gridding operations. Default to None,
        in which case as many grids as visibility chunks will
        be created.

    Returns
    -------
    grid : :class:`dask.array.Array`
        grid of shape :code:`(ny, nx, corr)`
    """
    if len(frequencies.chunks[0]) != 1:
        raise ValueError("Chunking in channel currently unsupported")

    # Create a baseline object per row chunk
    baselines = da.blockwise(_nifty_baselines, ("row",),
                             uvw, ("row", "uvw"),
                             frequencies, ("chan",),
                             dtype=object)

    if len(frequencies.chunks[0]) != 1:
        raise ValueError("Chunking in channel unsupported")

    gc = grid_config.object
    grids = []

    for corr in range(vis.shape[2]):
        corr_flags = flags[:, :, corr]
        corr_vis = vis[:, :, corr]
        corr_weights = weights[:, :, corr]

        indices = da.blockwise(_nifty_indices, ("row",),
                               baselines, ("row",),
                               gc, None,
                               corr_flags, ("row", "chan"),
                               -1, None,  # channel begin
                               -1, None,  # channel end
                               wmin, None,
                               wmax, None,
                               dtype=np.int32)

        if streams is None:
            # Standard parallel reduction, possibly memory hungry
            # if many threads (and thus grids) are gridding
            # parallel
            grid = da.blockwise(_nifty_grid, ("row", "nu", "nv"),
                                baselines, ("row",),
                                gc, None,
                                indices, ("row",),
                                corr_vis, ("row", "chan"),
                                corr_weights, ("row", "chan"),
                                new_axes={"nu": gc.Nu(), "nv": gc.Nv()},
                                adjust_chunks={"row": 1},
                                dtype=np.complex128)

            grids.append(grid.sum(axis=0))
        else:
            # Stream reduction
            layers = GridStreamReduction(baselines, indices, gc,
                                         corr_vis, corr_weights,
                                         corr, streams)
            deps = [baselines, indices, corr_vis, corr_weights]
            graph = HighLevelGraph.from_collections(layers.name, layers, deps)
            chunks = corr_vis.chunks
            grid_stream_red = da.Array(graph, layers.name, chunks, vis.dtype)

            layers = FinalGridReduction(layers)
            deps = [grid_stream_red]
            graph = HighLevelGraph.from_collections(layers.name, layers, deps)
            chunks = ((gc.Nu(),), (gc.Nv(),))
            corr_grid = da.Array(graph, layers.name, chunks, vis.dtype)

            grids.append(corr_grid)

    return da.stack(grids, axis=2)


def _nifty_dirty(grid, grid_config):
    """ Wrapper function for creating a dirty image """
    grids = [grid_config.grid2dirty_c(grid[:, :, c]).real
             for c in range(grid.shape[2])]

    return np.stack(grids, axis=2)


@requires_optional("dask.array", import_error)
@requires_optional("nifty_gridder", nifty_import_err)
def dirty(grid, grid_config):
    """
    Computes the dirty image from gridded visibilities and the
    gridding configuration.

    Parameters
    ----------
    grid : :class:`dask.array.Array`
        Gridded visibilities of shape :code:`(nv, nu, ncorr)`
    grid_config : :class:`GridderConfigWrapper`
        Gridding configuration

    Returns
    -------
    dirty : :class:`dask.array.Array`
        dirty image of shape :code:`(ny, nx, corr)`
    """

    gc = grid_config.object
    nx = gc.Nxdirty()
    ny = gc.Nydirty()

    return da.blockwise(_nifty_dirty, ("nx", "ny", "corr"),
                        grid, ("nx", "ny", "corr"),
                        gc, None,
                        adjust_chunks={"nx": nx, "ny": ny},
                        dtype=grid.real.dtype)


def _nifty_model(image, grid_config):
    """ Wrapper function for creating a dirty image """
    images = [grid_config.dirty2grid_c(image[:, :, c])
              for c in range(image.shape[2])]

    return np.stack(images, axis=2)


@requires_optional("dask.array", import_error)
@requires_optional("nifty_gridder", nifty_import_err)
def model(image, grid_config):
    """
    Computes model visibilities from an image
    and a gridding configuration.

    Parameters
    ----------
    image : :class:`dask.array.Array`
        Image of shape :code:`(ny, nx, corr)`.
    grid_config : :class:`GridderConfigWrapper`
        nifty gridding configuration object

    Returns
    -------
    model_vis : :class:`dask.array.Array`
        Model visibilities of shape :code:`(nu, nv, corr)`.
    """

    gc = grid_config.object
    nu = gc.Nu()
    nv = gc.Nv()

    return da.blockwise(_nifty_model, ("nu", "nv", "corr"),
                        image, ("nu", "nv", "corr"),
                        gc, None,
                        adjust_chunks={"nu": nu, "nv": nv},
                        dtype=image.dtype)


def _nifty_degrid(grid, baselines, indices, grid_config):
    assert len(grid) == 1 and len(grid[0]) == 1
    return ng.grid2ms_c(baselines, grid_config.object, indices, grid[0][0])


@requires_optional("dask.array", import_error)
@requires_optional("nifty_gridder", nifty_import_err)
def degrid(grid, uvw, flags, weights, frequencies,
           grid_config, wmin=-1e30, wmax=1e30):
    """
    Degrids the visibilities from the supplied grid in parallel.

    Parameters
    ----------
    grid : :class:`dask.array.Array`
        gridded visibilities of shape :code:`(ny, nx, corr)`
    uvw : :class:`dask.array.Array`
        uvw coordinates of shape :code:`(row, 3)`
    flags : :class:`dask.array.Array`
        flags of shape :code:`(row, chan, corr)`
    weights : :class:`dask.array.Array`
        weights of shape :code:`(row, chan, corr)`.
        Currently unsupported and ignored.
    frequencies : :class:`dask.array.Array`
        frequencies of shape :code:`(chan,)`
    grid_config : :class:`GridderConfigWrapper`
        Gridding Configuration
    wmin : float
        Minimum W coordinate to grid. Defaults to -1e30.
    wmax : float
        Maximum W coordinate to grid. Default to 1e30.

    Returns
    -------
    grid : :class:`dask.array.Array`
        grid of shape :code:`(ny, nx, corr)`
    """

    if len(frequencies.chunks[0]) != 1:
        raise ValueError("Chunking in channel currently unsupported")

    # Create a baseline object per row chunk
    baselines = da.blockwise(_nifty_baselines, ("row",),
                             uvw, ("row", "uvw"),
                             frequencies, ("chan",),
                             dtype=object)

    gc = grid_config.object
    vis_chunks = []

    for corr in range(grid.shape[2]):
        corr_flags = flags[:, :, corr].map_blocks(np.require, requirements="C")
        corr_grid = grid[:, :, corr].map_blocks(np.require, requirements="C")

        indices = da.blockwise(_nifty_indices, ("row",),
                               baselines, ("row",),
                               gc, None,
                               corr_flags, ("row", "chan"),
                               -1, None,  # channel begin
                               -1, None,  # channel end
                               wmin, None,
                               wmax, None,
                               dtype=np.int32)

        vis = da.blockwise(_nifty_degrid, ("row", "chan"),
                           corr_grid, ("ny", "nx"),
                           baselines, ("row",),
                           indices, ("row",),
                           grid_config, None,
                           new_axes={"chan": frequencies.shape[0]},
                           dtype=grid.dtype)

        vis_chunks.append(vis)

    return da.stack(vis_chunks, axis=2)
