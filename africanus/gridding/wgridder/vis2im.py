# -*- coding: utf-8 -*-

import numpy as np
from ducc0.wgridder import ms2dirty

def _hdot_wrapper(uvw, freq, ms, wgt, freq_bin_idx, freq_bin_counts,
                  nx, ny, cellx, celly, nu, nv, epsilon, nthreads, do_wstacking):
    
    return _hdot_internal(uvw[0][0], freq, ms[0][0], wgt[0][0],
                              freq_bin_idx, freq_bin_counts, nx, ny,
                              cellx, celly, nu, nv, epsilon, nthreads,
                              do_wstacking)

def _hdot_internal(uvw, freq, ms, wgt, freq_bin_idx, freq_bin_counts,
                   nx, ny, cellx, celly, nu, nv, epsilon, nthreads, do_wstacking):
    freq_bin_idx -= freq_bin_idx.min()  # adjust for chunking
    nband = freq_bin_idx.size
    dirty = np.zeros((nband, nx, ny), dtype=freq.dtype)
    for i in range(nband):
        I = slice(freq_bin_idx[i], freq_bin_idx[i] + freq_bin_counts[i])
        dirty[i] = ms2dirty(uvw=uvw, freq=freq[I], ms=ms[:, I, 0], wgt=wgt[:, I, 0], 
                            npix_x=nx, npix_y=ny, pixsize_x=cellx, pixsize_y=celly,
                            nu=nu, nv=nv, epsilon=epsilon, nthreads=nthreads, 
                            do_wstacking=do_wstacking) + \
                   ms2dirty(uvw=uvw, freq=freq[I], ms=ms[:, I, -1], wgt=wgt[:, I, -1], 
                            npix_x=nx, npix_y=ny, pixsize_x=cellx, pixsize_y=celly,
                            nu=nu, nv=nv, epsilon=epsilon, nthreads=nthreads, 
                            do_wstacking=do_wstacking)
    return dirty/2.0