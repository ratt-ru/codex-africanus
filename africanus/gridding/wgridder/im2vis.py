# -*- coding: utf-8 -*-

import numpy as np
from ducc0.wgridder import dirty2ms
def _dot_wrapper(uvw, freq, model, freq_bin_idx, freq_bin_counts,
                 cellx, celly, nu, nv, epsilon, nthreads, do_wstacking, ncorr, complex_type):
    return _dot_internal(uvw[0], freq, model[0][0],
                         freq_bin_idx, freq_bin_counts,
                         cellx, celly, nu, nv, epsilon, nthreads,
                         do_wstacking, ncorr, complex_type)

def _dot_internal(uvw, freq, model, freq_bin_idx, freq_bin_counts,
                  cellx, celly, nu, nv, epsilon, nthreads, do_wstacking, ncorr, complex_type):
    freq_bin_idx -= freq_bin_idx.min()  # adjust for chunking
    nband = freq_bin_idx.size
    nrow = uvw.shape[0]
    nchan = freq.size
    vis = np.zeros((nrow, nchan, ncorr), dtype=complex_type)
    for i in range(nband):
        I = slice(freq_bin_idx[i], freq_bin_idx[i] + freq_bin_counts[i])
        vis[:, I, 0] = dirty2ms(uvw=uvw, freq=freq[I], dirty=model[i], wgt=None, 
                                pixsize_x=cellx, pixsize_y=celly, nu=nu, nv=nv,
                                epsilon=epsilon, nthreads=nthreads, 
                                do_wstacking=do_wstacking)
        vis[:, I, -1] = vis[:, I, 0]  # assume no Stokes Q for now
    return vis