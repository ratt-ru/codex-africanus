import numpy as np
import copy
import time

try:
    # import scipy.stats as sstats
    # import scipy.signal as ssig
    import scipy.spatial as spat
except ImportError as e:
    opt_import_err = e
else:
    opt_import_err = None

from africanus.util.requirements import requires_optional


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print(("%r  %2.2f ms" % (method.__name__, (te - ts) * 1000)))

        return result

    return timed


class BoundingConvexHull(object):
    @requires_optional("scipy.stats", opt_import_err)
    def __init__(self, list_hulls, name="unnamed",
                 mask=None, check_mask_outofbounds=True):
        """
        Initializes a bounding convex hull around a list of bounding
        convex hulls or series of points.
        A unity-weighted mask is computed for the region that
        falls within this convex hull
        if a mask of (y, x) coordinates is not provided.
        Otherwise if a mask is provided and the
        check_mask_outofbounds value is set the
        masked coordinates are not verified to fall within
        the hull. The latter should thus be used with some
        caution by the user, but can potentially
        significantly speed up the mask creation process for
        axis aligned regions.
        """
        self._name = name
        self._check_mask_outofbounds = check_mask_outofbounds
        self._cached_filled_mask = None
        self._vertices = points = np.vstack(
            [
                b.corners if hasattr(b, "corners") else [b[0], b[1]]
                for b in list_hulls
            ]
        )
        self._hull = spat.ConvexHull(points)
        if mask is None:
            self._mask, self._mask_weights = self.init_mask()
        else:
            self.sparse_mask = mask

    def invalidate_cached_masks(self):
        """ Invalidates the cached masks (sparse or regular) """
        self._cached_filled_mask = None
        self._mask, self._mask_weights = self.init_mask()

    def __str__(self):
        return ",".join(
            ["({0:d},{1:d})".format(x, y) for (x, y) in self.corners]
        )

    def init_mask(self):
        """
        Creates a sparse mask of the convex hull of the form (y, x) tuples
        """
        lines = np.hstack([self.corners, np.roll(self.corners, -1, axis=0)])
        minx = np.min(lines[:, 0:4:2])
        maxx = np.max(lines[:, 0:4:2])
        miny = np.min(lines[:, 1:4:2])
        maxy = np.max(lines[:, 1:4:2])
        x = np.arange(minx, maxx + 1, 1)  # upper limit inclusive
        y = np.arange(miny, maxy + 1, 1)
        bounding_mesh = list(zip(*[np.ravel(x) for x in np.meshgrid(y, x)]))

        sparse_mask = (
            bounding_mesh
            if not self._check_mask_outofbounds
            else [c for c in bounding_mesh if c[::-1] in self]
        )

        # initialize to unity, this should be modified when coadding
        mask_weights = np.ones(len(sparse_mask))
        return sparse_mask, mask_weights

    @property
    def sprase_mask_weights(self):
        """ returns sparse mask weights """
        return self._mask_weights

    @property
    def sparse_mask(self):
        """
        Returns a sparse mask (y, x) values of all points in the masked region
        """
        return self._mask

    @sparse_mask.setter
    def sparse_mask(self, mask):
        """
        Sets the mask of the hull from
        a sparse mask - list of (y, x) coordinates
        """
        if not isinstance(mask, list):
            raise TypeError("Mask must be list")
        if not (
            hasattr(mask, "__len__")
            and (
                len(mask) == 0
                or (hasattr(mask[0], "__len__") and len(mask[0]) == 2)
            )
        ):
            raise TypeError("Mask must be a sparse mask of 2 element values")
        if self._check_mask_outofbounds:
            self._mask = copy.deepcopy(
                [c for c in mask if (c[1], c[0]) in self]
            )
        else:
            self._mask = copy.deepcopy(mask)
        self._mask_weights = np.ones(len(self._mask))

    @property
    def mask(self, dtype=np.float64):
        """ Creates a filled rectangular mask grid of size y, x """
        if self._cached_filled_mask is not None:
            return self._cached_filled_mask

        lines = np.hstack([self.corners, np.roll(self.corners, -1, axis=0)])
        minx = np.min(lines[:, 0:4:2])
        maxx = np.max(lines[:, 0:4:2])
        miny = np.min(lines[:, 1:4:2])
        maxy = np.max(lines[:, 1:4:2])
        nx = maxx - minx + 1  # inclusive
        ny = maxy - miny + 1
        mesh = np.zeros(nx * ny, dtype=dtype)
        if nx == 0 or ny == 0 or len(self.sparse_mask) == 0:
            self._cached_filled_mask = mesh.reshape((ny, nx))
        else:
            sparse_mask = np.array(self.sparse_mask)
            sel = np.logical_and(
                np.logical_and(
                    sparse_mask[:, 1] >= minx, sparse_mask[:, 1] <= maxx
                ),
                np.logical_and(
                    sparse_mask[:, 0] >= miny, sparse_mask[:, 0] <= maxy
                ),
            )
            flat_index = (sparse_mask[sel][:, 0] - miny) * nx + (
                sparse_mask[sel][:, 1] - minx
            )
            mesh[flat_index] = self._mask_weights[sel]
            self._cached_filled_mask = mesh.reshape((ny, nx))
        return self._cached_filled_mask

    @classmethod
    def regional_data(cls, sel_region, data_cube, axes=(2, 3), oob_value=0):
        """ 2D array containing all values within convex hull
            sliced out along axes provided as argument. Portions of sel_region
            that are outside of the data_cube is set to oob_value

            assumes the last value of axes is the fastest varying axis
        """
        if not isinstance(sel_region, BoundingConvexHull):
            raise TypeError(
                "Object passed in is not of type BoundingConvexHull"
            )
        if not (hasattr(axes, "__len__") and len(axes) == 2):
            raise ValueError(
                "Expected a tupple of axes along which to slice out a region"
            )
        axes = sorted(axes)

        lines = np.hstack(
            [sel_region.corners, np.roll(sel_region.corners, -1, axis=0)]
        )
        minx = np.min(lines[:, 0:4:2])
        maxx = np.max(lines[:, 0:4:2])
        miny = np.min(lines[:, 1:4:2])
        maxy = np.max(lines[:, 1:4:2])

        pad_left = max(0, 0 - minx)
        pad_bottom = max(0, 0 - miny)
        # inclusive of upper limit
        pad_right = max(0, maxx - data_cube.shape[axes[1]] + 1)
        pad_top = max(0, maxy - data_cube.shape[axes[0]] + 1)

        if (
            minx > data_cube.shape[axes[0]]
            or miny > data_cube.shape[axes[1]]
            or maxy < 0
            or maxx < 0
        ):
            raise ValueError("Expected a bounding hull that is "
                             "at least partially within the image")

        # extract data, pad if necessary
        slc_data = [slice(None)] * len(data_cube.shape)
        for (start, end), axis in zip(
            [
                (miny + pad_bottom, maxy - pad_top + 1),
                (minx + pad_left, maxx - pad_right + 1),
            ],
            axes,
        ):
            slc_data[axis] = slice(start, end)
        slc_padded = [slice(None)] * len(data_cube.shape)
        for (start, end), axis in zip(
            [
                (pad_bottom, -miny + maxy + 1 - pad_top),
                (pad_left, -minx + maxx + 1 - pad_right),
            ],
            axes,
        ):
            slc_padded[axis] = slice(start, end)

        selected_data = data_cube[tuple(slc_data)]
        new_shape = list(data_cube.shape)
        new_shape[axes[0]] = maxy - miny + 1
        new_shape[axes[1]] = maxx - minx + 1

        if any(np.array([pad_left, pad_bottom, pad_right, pad_top]) > 0):
            padded_data = (
                np.zeros(tuple(new_shape), dtype=selected_data.dtype)
                * oob_value
            )
            padded_data[tuple(slc_padded)] = selected_data.copy()
        else:
            padded_data = selected_data.copy()

        # finally apply mask
        slc_padded_data = [slice(None)] * len(padded_data.shape)
        for (start, end), axis in zip(
            [
                (
                    0,
                    maxy - miny + 1,
                ),  # mask starts at origin in the padded image
                (0, maxx - minx + 1),
            ],
            axes,
        ):
            slc_padded_data[axis] = slice(start, end)
        slc_mask = [None] * len(padded_data.shape)
        for (start, end), axis in zip(
            [
                (
                    0,
                    sel_region.mask.shape[0],
                ),  # mask starts at origin in the padded image
                (0, sel_region.mask.shape[1]),
            ],
            axes,
        ):
            slc_mask[axis] = slice(start, end)
        mask = sel_region.mask.copy()
        mask[mask == 0] = oob_value
        padded_data[tuple(slc_padded_data)] *= mask[tuple(slc_mask)]
        window_extents = [minx, maxx, miny, maxy]
        return padded_data, window_extents

    @classmethod
    def normalize_masks(cls, regions, only_overlapped_regions=True):
        """
        Normalizes region masks for overlapping pixels. This is necessary to
        properly coadd overlapping facets.
        If masks are guarenteed to be initialized to unity (e.g. after
        bounding region creation) the user can skip normalizing non-overlapping
        regions with flag only_overlapped_regions.
        """
        if not all([isinstance(reg, BoundingConvexHull) for reg in regions]):
            raise TypeError("Expected a list of bounding convex hulls")
        # Implements painters-like algorithm to
        # count the number of times a pixel coordinate falls within masks
        # The overlapping sections of regions can then be normalized
        # For now all regions have equal contribution
        allmasks = []
        for reg in regions:
            allmasks += (
                list(reg.sparse_mask)
                if isinstance(reg.sparse_mask, np.ndarray)
                else reg.sparse_mask
            )

        # flatten for faster comparisons
        allmasks = np.array(allmasks)
        maxx = np.max(allmasks[:, 1])
        nx = maxx + 1
        allmasks_flatten = allmasks[:, 0] * nx + allmasks[:, 1]

        # now count the number of times a pixel is painted onto
        unique_pxls_flatten, paint_count = np.unique(
            allmasks_flatten, return_counts=True
        )
        paint_count = paint_count.astype(float)

        if only_overlapped_regions:
            sel = paint_count > 1
            unique_pxls_flatten = unique_pxls_flatten[sel]
            paint_count = paint_count[sel]

        # with the reduced number of overlap pixels unflatten
        unique_pxls = np.vstack(
            [unique_pxls_flatten // nx, unique_pxls_flatten % nx]
        ).T
        unique_pxls = list(map(tuple, unique_pxls))
        paint_count[...] = 1.0 / paint_count

        # and finally update mask weights
        for reg in regions:
            reg._cached_filled_mask = None  # invalidate
            overlap = [
                x
                for x in zip(paint_count, unique_pxls)
                if x[1] in reg.sparse_mask
            ]
            for px_pc, px in overlap:
                sel = (
                    reg.sparse_mask.index(px)
                    if isinstance(reg.sparse_mask, list)
                    else np.all(reg.sparse_mask - px == 0, axis=1)
                )
                reg._mask_weights[sel] = px_pc

    @property
    def circumference(self):
        """ area contained in hull """
        lines = self.edges
        return np.sum(
            np.linalg.norm(lines[:, 1, :] - lines[:, 0, :], axis=1) + 1
        )

    @property
    def area(self):
        """ area contained in hull """
        lines = np.hstack([self.corners, np.roll(self.corners, -1, axis=0)])
        return (
            0.5
            * np.abs(
                np.sum([x1 * (y2) - (x2) * y1 for x1, y1, x2, y2 in lines])
            )
            + 0.5 * self.circumference
            - 1
        )

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, v):
        self._name = v

    @property
    def corners(self):
        """ Returns vertices and guarentees clockwise winding """
        return self._vertices[self._hull.vertices][::-1]

    def normals(self, left=True):
        """ return a list of left normals to the hull """
        normals = []
        for i in range(self.corners.shape[0]):
            # assuming clockwise winding
            j = (i + 1) % self.corners.shape[0]
            edge = self.corners[j, :] - self.corners[i, :]
            if left:
                normals.append((-edge[1], edge[0]))
            else:
                normals.append((edge[1], -edge[0]))
        return np.asarray(normals, dtype=np.double)

    @property
    def edges(self):
        """ return edge segments of the hull (clockwise wound) """
        edges = []
        for i in range(self.corners.shape[0]):
            # assuming clockwise winding
            j = (i + 1) % self.corners.shape[0]
            edge = tuple([self.corners[i, :], self.corners[j, :]])
            edges.append(edge)
        return np.asarray(edges, dtype=np.double)

    @property
    def edge_midpoints(self):
        """ return edge midpoints of the hull (clockwise wound) """
        edges = self.edges
        return np.mean(edges, axis=1)

    @property
    def lnormals(self):
        """ left normals to the edges of the hull """
        return self.normals(left=True)

    @property
    def rnormals(self):
        """ right normals to the edges of the hull """
        return self.normals(left=False)

    def overlaps_with(self, other, min_sep_dist=0.5):
        # less than half a pixel away
        """
        Implements the separating lines collision detection theorem
        to test whether the hull intersects with 'other' hull
        """
        if not isinstance(other, BoundingConvexHull):
            raise TypeError("rhs must be a BoundingConvexHull")

        # get the projection axes
        normals = np.vstack([self.lnormals, other.lnormals])
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / norms[None, 2]

        # compute vectors to corners from origin
        vecs_reg1 = self.corners
        vecs_reg2 = other.corners

        # compute projections onto normals
        for ni, n in enumerate(normals):
            projs = np.dot(vecs_reg1, n.T)
            minproj_reg1 = np.min(projs)
            maxproj_reg1 = np.max(projs)
            projs = np.dot(vecs_reg2, n.T)
            minproj_reg2 = np.min(projs)
            maxproj_reg2 = np.max(projs)
            if (
                minproj_reg2 - maxproj_reg1 > min_sep_dist
                or minproj_reg1 - maxproj_reg2 > min_sep_dist
            ):
                return False
        return True

    @property
    def centre(self, integral=True):
        """ Barycentre of hull """
        if integral:

            def rnd(x):
                return int(np.floor(x) if x >= 0 else np.ceil(x))

            return [rnd(x) for x in np.mean(self._vertices, axis=0)]
        else:
            return np.mean(self._vertices, axis=0)

    def __contains__(self, s, tolerance=0.5):  # less than half a pixel away
        """ tests whether a point s(x,y) is in the convex hull """
        # there are three cases to consider
        # CASE 1:
        # scalar projection  between all
        # inner pointing right normals (clockwise winding)
        # and the point must be positive if the point were to lie inside
        # the region (true)
        # CASE 2:
        # point is on an edge - the scalar projection onto
        # the axis is 0 for that edge
        # and greater than 0 for the other edges (true)
        # CASE 3:
        # it is outside (false)
        x, y = s
        normals = self.rnormals
        xyvec = np.array([x, y])[None, :] - np.array(self.corners)

        dot = np.einsum("ij,ij->i", normals, xyvec)
        return np.all(dot > -tolerance)


class BoundingBox(BoundingConvexHull):
    def __init__(self, xl, xu, yl, yu, name="unnamed", mask=None, **kwargs):
        if not all(
            [
                isinstance(x, (int, np.int64, np.int32, np.int16))
                for x in [xl, xu, yl, yu]
            ]
        ):
            raise ValueError("Box limits must be integers")
        self.__xnpx = abs(xu - xl + 1)  # inclusive of the upper pixel
        self.__ynpx = abs(yu - yl + 1)
        BoundingConvexHull.__init__(
            self,
            [[xl, yl], [xl, yu], [xu, yu], [xu, yl]],
            name,
            mask=mask,
            **kwargs
        )

    def init_mask(self):
        """
        creates a sparse mask of the convex hull of the form (y, x) tuples
        """
        lines = np.hstack([self.corners, np.roll(self.corners, -1, axis=0)])
        minx = np.min(lines[:, 0:4:2])
        maxx = np.max(lines[:, 0:4:2])
        miny = np.min(lines[:, 1:4:2])
        maxy = np.max(lines[:, 1:4:2])
        x = np.arange(minx, maxx + 1, 1)  # upper limit inclusive
        y = np.arange(miny, maxy + 1, 1)
        bounding_mesh = list(zip(*[np.ravel(x) for x in np.meshgrid(y, x)]))

        # by default for a BB region the mask is
        # always going to be the entire region
        sparse_mask = np.asarray(bounding_mesh)

        # initialize to unity, this should be modified when coadding
        mask_weights = np.ones(len(sparse_mask))
        return sparse_mask, mask_weights

    def __contains__(self, s):
        """ tests whether a point s(x,y) is in the box"""
        lines = np.hstack([self.corners, np.roll(self.corners, -1, axis=0)])
        minx = np.min(lines[:, 0:4:2])
        maxx = np.max(lines[:, 0:4:2])
        miny = np.min(lines[:, 1:4:2])
        maxy = np.max(lines[:, 1:4:2])
        return s[0] >= minx and s[0] <= maxx and s[1] >= miny and s[1] <= maxy

    @property
    def box_npx(self):
        return (self.__xnpx, self.__ynpx)

    @property
    def sparse_mask(self):
        """
        returns a sparse mask (y, x) values of all points in the masked region
        """
        return self._mask

    @sparse_mask.setter
    def sparse_mask(self, mask):
        """
        Sets the mask of the hull from a
        sparse mask - list of (y, x) coordinates
        """
        if not isinstance(mask, list) and not isinstance(mask, np.ndarray):
            raise TypeError("Mask must be list")
        if not (
            hasattr(mask, "__len__")
            and (
                len(mask) == 0
                or (hasattr(mask[0], "__len__") and len(mask[0]) == 2)
            )
        ):
            raise TypeError("Mask must be a sparse mask of 2 element values")
        if len(mask) == 0:
            self._mask = []
        else:
            lines = np.hstack(
                [self.corners, np.roll(self.corners, -1, axis=0)])
            minx = np.min(lines[:, 0:4:2])
            maxx = np.max(lines[:, 0:4:2])
            miny = np.min(lines[:, 1:4:2])
            maxy = np.max(lines[:, 1:4:2])
            sparse_mask = np.asarray(mask)
            sel = np.logical_and(
                np.logical_and(
                    sparse_mask[:, 1] >= minx, sparse_mask[:, 1] <= maxx
                ),
                np.logical_and(
                    sparse_mask[:, 0] >= miny, sparse_mask[:, 0] <= maxy
                ),
            )
            self._mask = sparse_mask[sel]
            self._mask_weights = np.ones(len(self._mask))

    @classmethod
    def project_regions(
        cls,
        regional_data_list,
        regions_list,
        axes=(2, 3),
        dtype=np.float64,
        **kwargs
    ):
        """ Projects individial regions back onto a single contiguous cube """
        if not (
            hasattr(regional_data_list, "__len__")
            and hasattr(regions_list, "__len__")
            and len(regions_list) == len(regional_data_list)
        ):
            raise TypeError("Region data list and regions lists "
                            "must be lists of equal length")
        if not all([isinstance(x, np.ndarray) for x in regional_data_list]):
            raise TypeError("Region data list must be a list of ndarrays")
        if not all([isinstance(x, BoundingBox) for x in regions_list]):
            raise TypeError(
                "Region list must be a list of Axis Aligned Bounding Boxes"
            )
        if regions_list == []:
            return np.empty((0))
        if not all(
            [
                reg.ndim == regional_data_list[0].ndim
                for reg in regional_data_list
            ]
        ):
            raise ValueError("All data cubes must be of equal dimension")
        axes = tuple(sorted(axes))

        minx = np.min([np.min(f.corners[:, 0]) for f in regions_list])
        maxx = np.max([np.max(f.corners[:, 0]) for f in regions_list])
        miny = np.min([np.min(f.corners[:, 1]) for f in regions_list])
        maxy = np.max([np.max(f.corners[:, 1]) for f in regions_list])
        npxx = maxx - minx + 1
        npxy = maxy - miny + 1
        global_offsetx = -minx  # -min(0, minx)
        global_offsety = -miny  # -min(0, miny)

        projected_image_size = list(regional_data_list[0].shape)
        projected_image_size[axes[0]] = npxy
        projected_image_size[axes[1]] = npxx
        stitched_img = np.zeros(tuple(projected_image_size), dtype=dtype)

        combined_mask = []
        for f, freg in zip(regional_data_list, regions_list):
            f[np.isnan(f)] = 0
            xl = max(0, global_offsetx + np.min(freg.corners[:, 0]))
            xu = min(global_offsetx + np.max(freg.corners[:, 0]) + 1, npxx)
            yl = max(0, global_offsety + np.min(freg.corners[:, 1]))
            yu = min(global_offsety + np.max(freg.corners[:, 1]) + 1, npxy)
            fnx = xu - xl + 1  # inclusive
            fny = yu - yl + 1  # inclusive
            if f.shape[axes[0]] != fny - 1 or f.shape[axes[1]] != fnx - 1:
                raise ValueError("One or more bounding box descriptors "
                                 "does not match shape of corresponding "
                                 "data cubes")
            slc_data = [slice(None)] * len(stitched_img.shape)
            for (start, end), axis in zip([(yl, yu), (xl, xu)], axes):
                slc_data[axis] = slice(start, end)

            stitched_img[tuple(slc_data)] += f
            combined_mask += list(freg.sparse_mask)

        return (
            stitched_img,
            BoundingBox(minx, maxx, miny, maxy, mask=combined_mask, **kwargs),
        )


########################################################################
# Factories
########################################################################


class BoundingBoxFactory(object):
    @classmethod
    def AxisAlignedBoundingBox(
        cls, convex_hull_object, square=False, enforce_odd=True, **kwargs
    ):
        """ Constructs an axis aligned bounding box around convex hull """
        if not isinstance(convex_hull_object, BoundingConvexHull):
            raise TypeError("Convex hull object passed in constructor "
                            "is not of type BoundingConvexHull")
        if square:
            nx = (
                np.max(convex_hull_object.corners[:, 0])
                - np.min(convex_hull_object.corners[:, 0])
                + 1
            )  # inclusive
            ny = (
                np.max(convex_hull_object.corners[:, 1])
                - np.min(convex_hull_object.corners[:, 1])
                + 1
            )  # inclusive
            boxdiam = max(nx, ny)
            boxrad = boxdiam // 2
            cx, cy = convex_hull_object.centre
            xl = cx - boxrad
            xu = cx + boxdiam - boxrad - 1
            yl = cy - boxrad
            yu = cy + boxdiam - boxrad - 1
        else:
            xl = np.min(convex_hull_object.corners[:, 0])
            xu = np.max(convex_hull_object.corners[:, 0])
            yl = np.min(convex_hull_object.corners[:, 1])
            yu = np.max(convex_hull_object.corners[:, 1])

        xu += (xu - xl) % 2 if enforce_odd else 0
        yu += (yu - yl) % 2 if enforce_odd else 0

        return BoundingBox(
            xl,
            xu,
            yl,
            yu,
            convex_hull_object.name,
            mask=convex_hull_object.sparse_mask,
            **kwargs
        )

    @classmethod
    def SplitBox(cls, bounding_box_object, nsubboxes=1, **kwargs):
        """ Split a axis-aligned bounding box into smaller boxes """
        if not isinstance(bounding_box_object, BoundingBox):
            raise TypeError("Expected bounding box object")
        if not (isinstance(nsubboxes, int) and nsubboxes >= 1):
            raise ValueError(
                "nsubboxes must be integral type and be 1 or more")
        xl = np.min(bounding_box_object.corners[:, 0])
        xu = np.max(bounding_box_object.corners[:, 0])
        yl = np.min(bounding_box_object.corners[:, 1])
        yu = np.max(bounding_box_object.corners[:, 1])

        # construct a nonregular meshgrid bound to xu and yu
        x = xl + np.arange(0, nsubboxes + 1) * int(
            np.ceil((xu - xl + 1) / float(nsubboxes))
        )
        y = yl + np.arange(0, nsubboxes + 1) * int(
            np.ceil((yu - yl + 1) / float(nsubboxes))
        )
        xx, yy = np.meshgrid(x, y)

        # split into boxes
        xls = xx[0:-1, 0:-1].copy()
        xus = xx[1:, 1:].copy()
        yls = yy[0:-1, 0:-1].copy()
        yus = yy[1:, 1:].copy()

        # make sure no boxes overlap
        xus = xus - 1
        yus = yus - 1

        # clamp the final coordinate to the upper end
        # (may result in rectanglular box at the end)
        xus[:, -1] = max(xu, min(xus[0, -1], xu))
        yus[-1, :] = max(yu, min(yus[-1, 0], yu))

        # coordinates for all the contained boxes, anti-clockwise wound
        xls = xls.ravel()
        yls = yls.ravel()
        xus = xus.ravel()
        yus = yus.ravel()
        bl = list(zip(xls, yls))
        br = list(zip(xus, yls))
        ur = list(zip(xus, yus))
        ul = list(zip(xls, yus))
        contained_boxes = list(zip(bl, br, ur, ul))

        # finally create bbs for each of the contained boxes with the mask
        # chopped up between the boxes by the convex hull initializer
        new_regions = [
            BoundingBox(
                bl[0],
                br[0],
                bl[1],
                ul[1],
                bounding_box_object.name,
                mask=bounding_box_object.sparse_mask,
                **kwargs
            )
            for bl, br, ur, ul in contained_boxes
        ]

        return new_regions

    @classmethod
    def PadBox(cls, bounding_box_object, desired_nx, desired_ny, **kwargs):
        """
        Creates a box with a padded border
        around a axis-aligned bounding box
        """
        if not isinstance(bounding_box_object, BoundingBox):
            raise TypeError("Expected bounding box object")
        nx, ny = bounding_box_object.box_npx
        if desired_nx - nx < 0 or desired_ny - ny < 0:
            raise ValueError("Padded box must be bigger than original box")
        pad_left = desired_nx // 2
        pad_right = desired_nx - pad_left - 1
        pad_bottom = desired_ny // 2
        pad_top = desired_ny - pad_bottom - 1
        cx, cy = bounding_box_object.centre
        xl = cx - pad_left
        xu = cx + pad_right
        yl = cy - pad_bottom
        yu = cy + pad_top
        return BoundingBox(
            xl,
            xu,
            yl,
            yu,
            bounding_box_object.name,
            mask=bounding_box_object.sparse_mask,
            **kwargs
        )  # mask unchanged in the new shape, border frame discarded
