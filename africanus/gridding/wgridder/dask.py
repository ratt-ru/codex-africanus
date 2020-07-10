# -*- coding: utf-8 -*-

import numpy as np

try:
    import dask
    import dask.array as da
except ImportError as e:
    import_error = e
else:
    import_error = None


def _dot_wrapper(uvw, freq, model, freq_bin_idx, freq_bin_counts,
                 cellx, celly, nu, nv, precision, nthreads, do_wstacking, ncorr, complex_type):

    vis = da.blockwise(_dot_wrapper, ('row', 'chan', 'corr'),
                                   uvw, ('row', 'three'), 
                                   freq, ('chan',),
                                   model, ('chan', 'nx', 'ny'),
                                   freq_bin_idx, ('chan',),
                                   freq_bin_counts, ('chan',),
                                   cellx, None, 
                                   celly, None,
                                   nu, None, 
                                   nv, None,
                                   precision, None,
                                   nthreads, None,
                                   do_wstacking, None,
                                   ncorr, None,
                                   complex_type, None,
                                   adjust_chunks={'chan': freq.chunks[0]},
                                   new_axes={"corr": ncorr},
                                   dtype=complex_type, 
                                   align_arrays=False)
    return _dot_internal(uvw[0], freq, model[0][0],
                         freq_bin_idx, freq_bin_counts,
                         cellx, celly, nu, nv, epsilon, nthreads,
                         do_wstacking, ncorr, complex_type)


def _hdot_wrapper(uvw, freq, ms, wgt, freq_bin_idx, freq_bin_counts,
                  nx, ny, cellx, celly, nu, nv, epsilon, nthreads, do_wstacking):
    
    return _hdot_internal(uvw[0][0], freq, ms[0][0], wgt[0][0],
                              freq_bin_idx, freq_bin_counts, nx, ny,
                              cellx, celly, nu, nv, epsilon, nthreads,
                              do_wstacking)