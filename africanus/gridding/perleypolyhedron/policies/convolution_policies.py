from africanus.util.numba import overload
import numpy as np
from . import stokes_conversion_policies as scp


def convolve_1d_axisymmetric_unpacked_scatter(
    scaled_u,
    scaled_v,
    scaled_w,
    npix,
    grid,
    vis,
    r,
    c,
    convolution_kernel,
    convolution_kernel_width,
    convolution_kernel_oversampling,
    stokes_conversion_policy,
    policy_type,
):
    """
    Convolution policy for a 1D axisymmetric unpacked
    AA kernel (gridding kernel)
    @scaled_u: simularity theorem and lambda scaled u
    @scaled_v: simularity theorem and lambda scaled v
    @scaled_w: simularity theorem and lambda scaled w
    @npix: number of pixels per axis
    @grid: 2d grid
    @r: current row in the visibility array
    @c: current channel in the visibility array
    @convolution_kernel: packed kernel as generated by
                         kernels package
    @convolution_kernel_width: number of taps in kernel
    @convolution_kernel_oversampling: number of oversampled
                                      points in kernel
    @stokes_conversion_policy: any accepted correlation to stokes
                               conversion policy in
                               .policies.stokes_conversion_policies
    """
    offset_u = scaled_u + npix // 2
    offset_v = scaled_v + npix // 2
    disc_u = int(np.round(offset_u))
    disc_v = int(np.round(offset_v))
    frac_u = int((-offset_u + disc_u) * convolution_kernel_oversampling)
    frac_v = int((-offset_v + disc_v) * convolution_kernel_oversampling)
    cw = 0.0
    for tv in range(convolution_kernel_width):
        conv_v = convolution_kernel[
            (tv + 1) * convolution_kernel_oversampling + frac_v
        ]
        grid_v_lookup = disc_v + tv - convolution_kernel_width // 2
        for tu in range(convolution_kernel_width):
            conv_u = convolution_kernel[
                (tu + 1) * convolution_kernel_oversampling + frac_u
            ]
            grid_u_lookup = disc_u + tu - convolution_kernel_width // 2
            if (
                grid_v_lookup >= 0
                and grid_v_lookup < npix
                and grid_u_lookup >= 0
                and grid_u_lookup < npix
            ):
                grid[grid_v_lookup, grid_u_lookup] += (
                    conv_v
                    * conv_u
                    * scp.corr2stokes(vis[r, c, :], stokes_conversion_policy)
                )
            cw += conv_v * conv_u
    return cw


def convolve_1d_axisymmetric_packed_scatter(
    scaled_u,
    scaled_v,
    scaled_w,
    npix,
    grid,
    vis,
    r,
    c,
    convolution_kernel,
    convolution_kernel_width,
    convolution_kernel_oversampling,
    stokes_conversion_policy,
    policy_type,
):
    """
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
    @stokes_conversion_policy: any accepted correlation to stokes
                               conversion policy in
                               .policies.stokes_conversion_policies
    """
    offset_u = scaled_u + npix // 2
    offset_v = scaled_v + npix // 2
    disc_u = int(np.round(offset_u))
    disc_v = int(np.round(offset_v))
    frac_u = int((-offset_u + disc_u) * convolution_kernel_oversampling)
    frac_v = int((-offset_v + disc_v) * convolution_kernel_oversampling)
    frac_offset_u = 0 if frac_u < 0 else +1
    frac_offset_v = 0 if frac_v < 0 else +1
    # Two cases here:
    # <--> padding before and
    #                 <--> after
    #     <---> half support
    # |...|...|...|...|...
    # 0123456789ABCDEFGHIJ
    # repacked into
    # 048CG159DH26AEI37BFJ
    # if fraction is negative we index 0 + frac * (support+2)
    # else we index 1 + frac * (support+2)
    # where frac wraps around to negative indexing
    cw = 0.0
    for tv in range(convolution_kernel_width):
        conv_v = convolution_kernel[
            tv + frac_offset_v + frac_v * (convolution_kernel_width + 2)
        ]
        grid_v_lookup = disc_v + tv - convolution_kernel_width // 2
        for tu in range(convolution_kernel_width):
            conv_u = convolution_kernel[
                tu + frac_offset_u + frac_u * (convolution_kernel_width + 2)
            ]
            grid_u_lookup = disc_u + tu - convolution_kernel_width // 2
            if (
                grid_v_lookup >= 0
                and grid_v_lookup < npix
                and grid_u_lookup >= 0
                and grid_u_lookup < npix
            ):
                grid[grid_v_lookup, grid_u_lookup] += (
                    conv_v
                    * conv_u
                    * scp.corr2stokes(vis[r, c, :], stokes_conversion_policy)
                )
            cw += conv_v * conv_u
    return cw


def convolve_nn_scatter(
    scaled_u,
    scaled_v,
    scaled_w,
    npix,
    grid,
    vis,
    r,
    c,
    convolution_kernel,
    convolution_kernel_width,
    convolution_kernel_oversampling,
    stokes_conversion_policy,
    policy_type,
):
    """
    Convolution policy for a nn scatter kernel (gridding kernel)
    @scaled_u: simularity theorem and lambda scaled u
    @scaled_v: simularity theorem and lambda scaled v
    @scaled_w: simularity theorem and lambda scaled w
    @npix: number of pixels per axis
    @grid: 2d grid
    @r: current row in the visibility array
    @c: current channel in the visibility array
    @convolution_kernel: packed kernel as generated by
                         kernels package
    @convolution_kernel_width: number of taps in kernel
    @convolution_kernel_oversampling: number of oversampled
                                      points in kernel
    @stokes_conversion_policy: any accepted correlation to stokes
                               conversion policy in
                               .policies.stokes_conversion_policies
    """
    offset_u = scaled_u + npix // 2
    offset_v = scaled_v + npix // 2
    disc_u = int(np.round(offset_u))
    disc_v = int(np.round(offset_v))
    cw = 1.0
    grid[disc_v, disc_u] += scp.corr2stokes(
        vis[r, c, :], stokes_conversion_policy
    )
    return cw


def convolve_1d_axisymmetric_packed_gather(
    scaled_u,
    scaled_v,
    scaled_w,
    npix,
    grid,
    vis,
    r,
    c,
    convolution_kernel,
    convolution_kernel_width,
    convolution_kernel_oversampling,
    stokes_conversion_policy,
    policy_type,
):
    """
    Convolution policy for a 1D axisymmetric packed AA
    kernel (degridding kernel)
    @scaled_u: simularity theorem and lambda scaled u
    @scaled_v: simularity theorem and lambda scaled v
    @scaled_w: simularity theorem and lambda scaled w
    @npix: number of pixels per axis
    @grid: 2d grid
    @r: current row in the visibility array
    @c: current channel in the visibility array
    @convolution_kernel: packed kernel as generated by
                         kernels package
    @convolution_kernel_width: number of taps in kernel
    @convolution_kernel_oversampling: number of oversampled points in kernel
    @stokes_conversion_policy: any accepted correlation to
                               stokes conversion policy in
                               .policies.stokes_conversion_policies
    """
    offset_u = scaled_u + npix // 2
    offset_v = scaled_v + npix // 2
    disc_u = int(np.round(offset_u))
    disc_v = int(np.round(offset_v))
    frac_u = int((-offset_u + disc_u) * convolution_kernel_oversampling)
    frac_v = int((-offset_v + disc_v) * convolution_kernel_oversampling)
    frac_offset_u = 0 if frac_u < 0 else +1
    frac_offset_v = 0 if frac_v < 0 else +1
    # Two cases here:
    # <--> padding before and
    #                 <--> after
    #     <---> half support
    # |...|...|...|...|...
    # 0123456789ABCDEFGHIJ
    # repacked into
    # 048CG159DH26AEI37BFJ
    # if fraction is negative we index 0 + frac * (support+2)
    # else we index 1 + frac * (support+2)
    # where frac wraps around to negative indexing
    cw = 0
    for tv in range(convolution_kernel_width):
        conv_v = convolution_kernel[
            tv + frac_offset_v + frac_v * (convolution_kernel_width + 2)
        ]
        grid_v_lookup = disc_v + tv - convolution_kernel_width // 2
        for tu in range(convolution_kernel_width):
            conv_u = convolution_kernel[
                tu + frac_offset_u + frac_u * (convolution_kernel_width + 2)
            ]
            grid_u_lookup = disc_u + tu - convolution_kernel_width // 2
            if (
                grid_v_lookup >= 0
                and grid_v_lookup < npix
                and grid_u_lookup >= 0
                and grid_u_lookup < npix
            ):
                scp.stokes2corr(
                    grid[
                        disc_v + tv - convolution_kernel_width // 2,
                        disc_u + tu - convolution_kernel_width // 2,
                    ]
                    * conv_v
                    * conv_u,
                    vis[r, c, :],
                    policy_type=stokes_conversion_policy,
                )
                cw += conv_v * conv_u
    vis[r, c, :] /= cw + 1.0e-8


def convolve_1d_axisymmetric_unpacked_gather(
    scaled_u,
    scaled_v,
    scaled_w,
    npix,
    grid,
    vis,
    r,
    c,
    convolution_kernel,
    convolution_kernel_width,
    convolution_kernel_oversampling,
    stokes_conversion_policy,
    policy_type,
):
    """
    Convolution policy for a 1D axisymmetric unpacked
    AA kernel (degridding kernel)
    @scaled_u: simularity theorem and lambda scaled u
    @scaled_v: simularity theorem and lambda scaled v
    @scaled_w: simularity theorem and lambda scaled w
    @npix: number of pixels per axis
    @grid: 2d grid
    @r: current row in the visibility array
    @c: current channel in the visibility array
    @convolution_kernel: packed kernel as generated by kernels package
    @convolution_kernel_width: number of taps in kernel
    @convolution_kernel_oversampling: number of oversampled points
                                      in kernel
    @stokes_conversion_policy: any accepted correlation to stokes
                               conversion policy in
                               .policies.stokes_conversion_policies
    """
    offset_u = scaled_u + npix // 2
    offset_v = scaled_v + npix // 2
    disc_u = int(np.round(offset_u))
    disc_v = int(np.round(offset_v))
    frac_u = int((-offset_u + disc_u) * convolution_kernel_oversampling)
    frac_v = int((-offset_v + disc_v) * convolution_kernel_oversampling)
    cw = 0
    for tv in range(convolution_kernel_width):
        conv_v = convolution_kernel[
            (tv + 1) * convolution_kernel_oversampling + frac_v
        ]
        grid_v_lookup = disc_v + tv - convolution_kernel_width // 2
        for tu in range(convolution_kernel_width):
            conv_u = convolution_kernel[
                (tu + 1) * convolution_kernel_oversampling + frac_u
            ]
            grid_u_lookup = disc_u + tu - convolution_kernel_width // 2
            if (
                grid_v_lookup >= 0
                and grid_v_lookup < npix
                and grid_u_lookup >= 0
                and grid_u_lookup < npix
            ):
                scp.stokes2corr(
                    grid[grid_v_lookup, grid_u_lookup] * conv_v * conv_u,
                    vis[r, c, :],
                    policy_type=stokes_conversion_policy,
                )
                cw += conv_v * conv_u
    vis[r, c, :] /= cw + 1.0e-8


def policy(
    scaled_u,
    scaled_v,
    scaled_w,
    npix,
    grid,
    vis,
    r,
    c,
    convolution_kernel,
    convolution_kernel_width,
    convolution_kernel_oversampling,
    stokes_conversion_policy,
    policy_type,
):
    pass


@overload(policy, inline="always")
def policy_impl(
    scaled_u,
    scaled_v,
    scaled_w,
    npix,
    grid,
    vis,
    r,
    c,
    convolution_kernel,
    convolution_kernel_width,
    convolution_kernel_oversampling,
    stokes_conversion_policy,
    policy_type,
):
    if policy_type.literal_value == "conv_1d_axisymmetric_packed_scatter":
        return convolve_1d_axisymmetric_packed_scatter
    elif policy_type.literal_value == "conv_nn_scatter":
        return convolve_nn_scatter
    elif policy_type.literal_value == "conv_1d_axisymmetric_unpacked_scatter":
        return convolve_1d_axisymmetric_unpacked_scatter
    elif policy_type.literal_value == "conv_1d_axisymmetric_packed_gather":
        return convolve_1d_axisymmetric_packed_gather
    elif policy_type.literal_value == "conv_1d_axisymmetric_unpacked_gather":
        return convolve_1d_axisymmetric_unpacked_gather
    else:
        raise ValueError("Invalid convolution policy type")
