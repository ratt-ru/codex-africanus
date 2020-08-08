import numpy as np
from numba import jit, literally, prange
from .policies import baseline_transform_policies as btp
from .policies import phase_transform_policies as ptp
from .policies import convolution_policies as cp
from .policies import stokes_conversion_policies as scp

@jit(nopython=True,nogil=True,fastmath=True,parallel=True)
def degridder(uvw,
              gridstack,
              lambdas,
              chanmap,
              cell,
              image_centre,
              phase_centre,
              convolution_kernel,
              convolution_kernel_width,
              convolution_kernel_oversampling,
              baseline_transform_policy,
              phase_transform_policy,
              stokes_conversion_policy,
              convolution_policy,
              vis_dtype=np.complex128):
    """
    2D Convolutional degridder, discrete to contiguous
    @uvw: value coordinates, (nrow, 3)
    @gridstack: complex gridded data, (nband, npix, npix)
    @lambdas: wavelengths of data channels
    @chanmap: MFS band mapping per channel
    @cell: cell_size in degrees
    @image_centre: new phase centre of image (radians, ra, dec)
    @phase_centre: original phase centre of data (radians, ra, dec)
    @convolution_kernel: packed kernel as generated by kernels package
    @convolution_kernel_width: number of taps in kernel
    @convolution_kernel_oversampling: number of oversampled points in kernel
    @baseline_transform_policy: any accepted policy in .policies.baseline_transform_policies,
                                can be used to tilt image planes for polyhedron faceting
    @phase_transform_policy: any accepted policy in .policies.phase_transform_policies,
                             can be used to facet at provided facet @image_centre
    @stokes_conversion_policy: any accepted correlation to stokes conversion policy in
                               .policies.stokes_conversion_policies
    @convolution_policy: any accepted convolution policy in
                         .policies.convolution_policies
    @vis_dtype: accumulation vis dtype (default complex 128)
    """

    if chanmap.size != lambdas.size:
        raise ValueError("Chanmap and corresponding lambdas must match in shape")
    chanmap = chanmap.ravel()
    lambdas = lambdas.ravel()
    nband = np.max(chanmap) + 1
    nrow = uvw.shape[0]
    npix = gridstack.shape[1]
    if gridstack.shape[1] != gridstack.shape[2]:
        raise ValueError("Grid must be square")
    nvischan = lambdas.size
    ncorr = scp.ncorr_out(policy_type=literally(stokes_conversion_policy))
    if gridstack.shape[0] < nband:
        raise ValueError("Not enough channel bands in grid stack to match mfs band mapping")
    if uvw.shape[1] != 3:
        raise ValueError("UVW array must be array of tripples")
    if uvw.shape[0] != nrow:
        raise ValueError("UVW array must have same number of rows as vis array")
    if nvischan != lambdas.size:
        raise ValueError("Chanmap must correspond to visibility channels")
    
    vis = np.zeros((nrow, nvischan, ncorr), dtype=vis_dtype)

    # scale the FOV using the simularity theorem
    scale_factor = npix * cell / 3600.0 * np.pi / 180.0
    for r in prange(nrow):
        ra0, dec0 = phase_centre
        ra, dec = image_centre
        btp.policy(uvw[r,:], ra, dec, ra0, dec0, literally(baseline_transform_policy))
        for c in range(nvischan):
            scaled_u = uvw[r,0] * scale_factor / lambdas[c]
            scaled_v = uvw[r,1] * scale_factor / lambdas[c]
            scaled_w = uvw[r,2] * scale_factor / lambdas[c]
            grid = gridstack[chanmap[c],:,:]
            cp.policy(scaled_u, scaled_v, scaled_w, 
                      npix, grid, vis, r, c,
                      convolution_kernel,
                      convolution_kernel_width,
                      convolution_kernel_oversampling,
                      stokes_conversion_policy,
                      policy_type=literally(convolution_policy))
        ptp.policy(vis[r,:,:], uvw[r,:], lambdas,
                   ra0, dec0, ra, dec, 
                   policy_type=literally(phase_transform_policy), 
                   phasesign=-1.0)
    return vis