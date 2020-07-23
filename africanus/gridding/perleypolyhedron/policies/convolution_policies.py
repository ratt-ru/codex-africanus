from numba import jit, float32, float64, literally
from numba.extending import overload
import numpy as np
from . import stokes_conversion_policies as scp

def convolve_1d_axisymmetric_packed_scatter(scaled_u, scaled_v, scaled_w,
                                            npix,
                                            grid,
                                            vis,
                                            r,c,
                                            convolution_kernel,
                                            convolution_kernel_width,
                                            convolution_kernel_oversampling,
                                            stokes_conversion_policy,
                                            policy_type):
    '''
    Convolution policy for a 1D axisymmetric packed AA kernel (gridding kernel)
    @scaled_u: simularity theorem and lambda scaled u
    @scaled_v: simularity theorem and lambda scaled v
    @scaled_w: simularity theorem and lambda scaled w
    @npix: number of pixels per axis
    @grid: 2d grid
    @r: current row in the visibility array
    @c: current channel in the visibility array
    @convolution_kernel: packed kernel as generated by kernels package
    @convolution_kernel_width: number of taps in kernel
    @convolution_kernel_oversampling: number of oversampled points in kernel
    @stokes_conversion_policy: any accepted correlation to stokes conversion policy in
                               .policies.stokes_conversion_policies
    '''
    offset_u = scaled_u + npix // 2
    offset_v = scaled_v + npix // 2
    disc_u = int(np.round(offset_u))
    disc_v = int(np.round(offset_v))
    frac_u = int((offset_u - disc_u) * convolution_kernel_oversampling)
    if frac_u < 0:
        frac_u = int((1.0 - (offset_u - disc_u)) * convolution_kernel_oversampling)
    frac_v = int((offset_v - disc_v) * convolution_kernel_oversampling)
    if frac_v < 0:
        frac_v = int((1.0 - (offset_v - disc_v)) * convolution_kernel_oversampling)
    # by definition the fraction is positive
    # assume unpacked taps are specified as
    # 0   1   2   3   4   = W
    # |...|...|...|...|...
    # then use int(frac_u * oversample) * W + n for n in [0..W)
    # in the packed taps
    for tv in range(convolution_kernel_width):
        conv_v = convolution_kernel[tv + frac_v * convolution_kernel_width]
        grid_v_lookup = disc_v + tv - convolution_kernel_width//2
        for tu in range(convolution_kernel_width):
            conv_u = convolution_kernel[tu + frac_u * convolution_kernel_width]
            grid_u_lookup = disc_u + tu - convolution_kernel_width//2
            if (grid_v_lookup >= 0 and grid_v_lookup < npix and 
                grid_u_lookup >= 0 and grid_u_lookup < npix):
                grid[disc_v + tv - convolution_kernel_width//2, 
                        disc_u + tu - convolution_kernel_width//2] += \
                            conv_v * conv_u * scp.corr2stokes(vis[r,c,:], 
                                                            literally(stokes_conversion_policy))

def convolve_1d_axisymmetric_packed_gather(scaled_u, scaled_v, scaled_w,
                                           npix,
                                           grid,
                                           vis,
                                           r,c,
                                           convolution_kernel,
                                           convolution_kernel_width,
                                           convolution_kernel_oversampling,
                                           stokes_conversion_policy,
                                           policy_type):
    '''
    Convolution policy for a 1D axisymmetric packed AA kernel (degridding kernel)
    @scaled_u: simularity theorem and lambda scaled u
    @scaled_v: simularity theorem and lambda scaled v
    @scaled_w: simularity theorem and lambda scaled w
    @npix: number of pixels per axis
    @grid: 2d grid
    @r: current row in the visibility array
    @c: current channel in the visibility array
    @convolution_kernel: packed kernel as generated by kernels package
    @convolution_kernel_width: number of taps in kernel
    @convolution_kernel_oversampling: number of oversampled points in kernel
    @stokes_conversion_policy: any accepted correlation to stokes conversion policy in
                               .policies.stokes_conversion_policies
    '''
    offset_u = scaled_u + npix // 2
    offset_v = scaled_v + npix // 2
    disc_u = int(np.round(offset_u))
    disc_v = int(np.round(offset_v))
    frac_u = int((offset_u - disc_u) * convolution_kernel_oversampling)
    if frac_u < 0:
        frac_u = int((1.0 - (offset_u - disc_u)) * convolution_kernel_oversampling)
    frac_v = int((offset_v - disc_v) * convolution_kernel_oversampling)
    if frac_v < 0:
        frac_v = int((1.0 - (offset_v - disc_v)) * convolution_kernel_oversampling)
    # by definition the fraction is positive
    # assume unpacked taps are specified as
    # 0   1   2   3   4   = W
    # |...|...|...|...|...
    # then use int(frac_u * oversample) * W + n for n in [0..W)
    # in the packed taps
    cw = 0
    for tv in range(convolution_kernel_width):
        conv_v = convolution_kernel[tv + frac_v * convolution_kernel_width]
        grid_v_lookup = disc_v + tv - convolution_kernel_width//2
        for tu in range(convolution_kernel_width):
            conv_u = convolution_kernel[tu + frac_u * convolution_kernel_width]
            grid_u_lookup = disc_u + tu - convolution_kernel_width//2
            if (grid_v_lookup >= 0 and grid_v_lookup < npix and 
                grid_u_lookup >= 0 and grid_u_lookup < npix):
                scp.stokes2corr(grid[disc_v + tv - convolution_kernel_width//2, 
                                     disc_u + tu - convolution_kernel_width//2] * conv_v * conv_u,
                                vis[r,c,:],
                                policy_type=literally(stokes_conversion_policy))
                cw += conv_v * conv_u
    vis[r,c,:] /= cw + 1.0e-8
def policy(scaled_u, scaled_v, scaled_w,
           npix,
           grid,
           vis,
           r,c,
           convolution_kernel,
           convolution_kernel_width,
           convolution_kernel_oversampling,
           stokes_conversion_policy,
           policy_type):
    pass

@overload(policy, inline="always")
def policy_impl(scaled_u, scaled_v, scaled_w,
                npix,
                grid,
                vis,
                r,c,
                convolution_kernel,
                convolution_kernel_width,
                convolution_kernel_oversampling,
                stokes_conversion_policy,
                policy_type):
    if policy_type.literal_value == "conv_1d_axisymmetric_packed_scatter":
        return convolve_1d_axisymmetric_packed_scatter
    elif policy_type.literal_value == "conv_1d_axisymmetric_packed_gather":
        return convolve_1d_axisymmetric_packed_gather   
    else:
        raise ValueError("Invalid convolution policy type")