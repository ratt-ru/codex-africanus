import dask
import dask.array as da
import numpy as np

from africanus.gridding.perleypolyhedron.gridder import gridder as np_gridder

def __grid(uvw,
           vis,
           image_centres,
           lambdas=None,
           chanmap=None,
           convolution_kernel=None,
           convolution_kernel_width=None,
           convolution_kernel_oversampling=None,
           baseline_transform_policy=None,
           phase_transform_policy=None,
           stokes_conversion_policy=None,
           convolution_policy=None,
           npix=None,
           cell=None,
           phase_centre=None,
           grid_dtype=np.complex128,
           do_normalize=False
    ):
    image_centres = image_centres[0]
    if image_centres.ndim != 2:
        raise ValueError("Image centres for DASK wrapper expects list of image centres, one per facet in radec radians")
    if image_centres.shape[1] != 2:
        raise ValueError("Image centre must be a list of tuples")

    uvw = uvw[0]
    vis = vis[0][0]
    lambdas = lambdas[0]
    chanmap = chanmap[0]
    grid_stack = np.zeros((1, image_centres.shape[0], 1, np.max(chanmap) + 1, npix, npix),
                          dtype=grid_dtype)
    for fi, f in enumerate(image_centres):
        grid_stack[0, fi, 0, :, :, :] = \
            np_gridder(uvw, vis, lambdas, chanmap, npix, cell, f, phase_centre,
                    convolution_kernel, convolution_kernel_width, convolution_kernel_oversampling,
                    baseline_transform_policy, phase_transform_policy, stokes_conversion_policy,
                    convolution_policy, grid_dtype, do_normalize)
    return grid_stack

def gridder(uvw,
            vis,
            lambdas,
            chanmap,
            npix,
            cell,
            image_centres,
            phase_centre,
            convolution_kernel,
            convolution_kernel_width,
            convolution_kernel_oversampling,
            baseline_transform_policy,
            phase_transform_policy,
            stokes_conversion_policy,
            convolution_policy,
            grid_dtype=np.complex128,
            do_normalize=False):
    """
    2D Convolutional gridder, contiguous to discrete
    @uvw: value coordinates, (nrow, 3)
    @vis: complex data, (nrow, nchan, ncorr)
    @lambdas: wavelengths of data channels
    @chanmap: MFS band mapping
    @npix: number of pixels per axis
    @cell: cell_size in degrees
    @image_centres: new phase centres of images (nfacet, (radians ra, dec))
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
    @grid_dtype: accumulation grid dtype (default complex 128)
    @do_normalize: normalize grid by convolution weights
    """
    if len(vis.chunks) != 3 or lambdas.chunks[0] != vis.chunks[1]:
        raise ValueError("Visibility frequency chunking does not match lambda frequency chunking")
    if len(vis.chunks) != 3 or chanmap.chunks[0] != vis.chunks[1]:
        raise ValueError("Visibility frequency chunking does not match chanmap frequency chunking")
    if len(vis.chunks) != 3 or len(uvw.chunks) != 2 or vis.chunks[0] != uvw.chunks[0]:
        raise ValueError("Visibility row chunking does not match uvw row chunking")
    grids = da.blockwise(__grid, ("row", "nfacet", "nstokes", "nband", "y", "x"),
                        uvw, ("row", "uvw"),
                        vis, ("row", "chan", "corr"),
                        image_centres, ("nfacet", "coord"),
                        lambdas, ("chan",),
                        chanmap, ("chan",),
                        convolution_kernel=convolution_kernel,
                        convolution_kernel_width=convolution_kernel_width,
                        convolution_kernel_oversampling=convolution_kernel_oversampling,
                        baseline_transform_policy=baseline_transform_policy,
                        phase_transform_policy=phase_transform_policy,
                        stokes_conversion_policy=stokes_conversion_policy,
                        convolution_policy=convolution_policy,
                        npix=npix,
                        cell=cell,
                        phase_centre=phase_centre,
                        grid_dtype=grid_dtype,
                        do_normalize=do_normalize,
                        adjust_chunks={"row": 1}, # goes to one set of grids per row chunk
                        new_axes={"nband": np.max(chanmap) + 1,
                                  "nstokes": 1, # for now will need to be modified if multi-stokes cubes are supported
                                  "y": npix,
                                  "x": npix},
                        dtype=grid_dtype,
                        meta=np.empty((0,0,0,0,0,0), dtype=grid_dtype)
                        )

    # Parallel reduction over row dimension
    return grids.mean(axis=0, split_every=2)