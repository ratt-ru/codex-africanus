# -*- coding: utf-8 -*-

import numpy as np

try:
    import dask
    import dask.array as da
except ImportError as e:
    import_error = e
else:
    import_error = None

from africanus.gridding.wgridder.vis2im import VIS2IM_DOCS
from africanus.gridding.wgridder.im2vis import IM2VIS_DOCS
from africanus.gridding.wgridder.im2residim import IM2RESIDIM_DOCS
from africanus.gridding.wgridder import im2vis as im2vis_np
from africanus.gridding.wgridder import vis2im as vis2im_np
from africanus.gridding.wgridder import im2residim as im2residim_np


def _im2vis_wrapper(uvw, freq, model, weights, freq_bin_idx, freq_bin_counts,
                    cellx, celly, nu, nv, epsilon, nthreads, do_wstacking, complex_type):

    return im2vis_np(uvw[0], freq, model[0][0], weights, freq_bin_idx,
                    freq_bin_counts, cellx, celly, nu, nv,
                    epsilon, nthreads, do_wstacking, complex_type)

def im2vis(uvw, freq, model, weights, freq_bin_idx, freq_bin_counts,
           cellx, celly, nu, nv, epsilon, nthreads, do_wstacking, complex_type):
    vis = da.blockwise(_im2vis_wrapper, ('row', 'chan'),
                    uvw, ('row', 'three'), 
                    freq, ('chan',),
                    model, ('chan', 'nx', 'ny'),
                    weights, ('row', 'chan'),
                    freq_bin_idx, ('chan',),
                    freq_bin_counts, ('chan',),
                    cellx, None, 
                    celly, None,
                    nu, None, 
                    nv, None,
                    epsilon, None,
                    nthreads, None,
                    do_wstacking, None,
                    complex_type, None,
                    adjust_chunks={'chan': freq.chunks[0]},
                    dtype=complex_type, 
                    align_arrays=False)
    return vis


def _vis2im_wrapper(uvw, freq, vis, weights, freq_bin_idx, freq_bin_counts,
                    nx, ny, cellx, celly, nu, nv, epsilon, nthreads,
                    do_wstacking):

    return vis2im_np(uvw[0][0], freq, vis[0], weights[0], freq_bin_idx,
                     freq_bin_counts, nx, ny, cellx, celly, nu, nv,
                     epsilon, nthreads, do_wstacking)

def vis2im(uvw, freq, vis, weights, freq_bin_idx, freq_bin_counts,
            nx, ny, cellx, celly, nu, nv, epsilon, nthreads, do_wstacking):

    dirty = da.blockwise(_vis2im_wrapper, ('chan', 'nx', 'ny'),
                        uvw, ('row', 'three'), 
                        freq, ('chan',),
                        vis, ('row', 'chan'),
                        weights, ('row', 'chan'),
                        freq_bin_idx, ('chan',),
                        freq_bin_counts, ('chan',),
                        nx, None,
                        ny, None, 
                        cellx, None, 
                        celly, None,
                        nu, None, 
                        nv, None,
                        epsilon, None,
                        nthreads, None,
                        do_wstacking, None,
                        adjust_chunks={'chan': freq_bin_idx.chunks[0]},
                        new_axes={"nx": nx, "ny": ny},
                        dtype=weights.dtype, 
                        align_arrays=False)
    
    return dirty

def _im2residim_wrapper(uvw, freq, model, vis, weights, freq_bin_idx,
                        freq_bin_counts, cellx, celly, nu, nv, epsilon,
                        nthreads, do_wstacking):

    return im2residim_np(uvw[0], freq, model, vis[0], weights[0],
                        freq_bin_idx, freq_bin_counts, cellx, celly, nu, nv,
                        epsilon, nthreads, do_wstacking)

def im2residim(uvw, freq, model, vis, weights, freq_bin_idx, freq_bin_counts,
               cellx, celly, nu, nv, epsilon, nthreads, do_wstacking):

    residim = da.blockwise(_im2residim_wrapper, ('chan', 'nx', 'ny'),
                        uvw, ('row', 'three'), 
                        freq, ('chan',),
                        model, ('chan', 'nx', 'ny'),
                        vis, ('row', 'chan'),
                        weights, ('row', 'chan'),
                        freq_bin_idx, ('chan',),
                        freq_bin_counts, ('chan',),
                        cellx, None, 
                        celly, None,
                        nu, None, 
                        nv, None,
                        epsilon, None,
                        nthreads, None,
                        do_wstacking, None,
                        adjust_chunks={'chan': freq.chunks[0]},
                        dtype=model.dtype, 
                        align_arrays=False)
    return residim

im2vis.__doc__ = IM2VIS_DOCS.substitute(
                    array_type=":class:`dask.array.Array`")
vis2im.__doc__ = VIS2IM_DOCS.substitute(
                    array_type=":class:`dask.array.Array`")
im2residim.__doc__ = IM2RESIDIM_DOCS.substitute(
                            array_type=":class:`dask.array.Array`")