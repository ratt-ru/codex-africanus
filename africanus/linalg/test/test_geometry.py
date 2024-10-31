import numpy as np
import pytest

from africanus.linalg.geometry import (
    BoundingConvexHull,
    BoundingBox,
    BoundingBoxFactory,
)


@pytest.mark.parametrize("debug", [False])
def test_hull_construction(debug):
    # test case 1
    vals = np.array([[50, 60], [20, 40], [-74, 50], [-95, +10], [20, 60]])
    bh = BoundingConvexHull(vals)
    mask = bh.mask
    assert mask.shape == (
        np.max(vals[:, 1]) - np.min(vals[:, 1]) + 1,
        np.max(vals[:, 0]) - np.min(vals[:, 0]) + 1,
    )
    # integral mask area needs to be close to true area
    assert np.abs(mask.sum() - bh.area) / bh.area < 0.05
    normalized_normals = bh.rnormals / np.linalg.norm(bh.rnormals, axis=1)[:, None]
    # test case 2
    for e, n in zip(bh.edges, normalized_normals):
        edge_vec = e[1] - e[0]
        assert np.all(np.abs(np.dot(edge_vec, n)) < 1.0e-8)

    # test case 3
    valsextract = np.array([[-10, 120], [90, 268], [293, 110], [40, -30]])
    bh_extract = BoundingConvexHull(valsextract)
    sinc_npx = 255
    sinc = np.sinc(np.linspace(-7, 7, sinc_npx))
    sinc2d = np.outer(sinc, sinc).reshape((1, 1, sinc_npx, sinc_npx))
    (extracted_data, extracted_window_extents) = BoundingConvexHull.regional_data(
        bh_extract, sinc2d, oob_value=np.nan
    )
    assert extracted_window_extents == [-10, 293, -30, 268]
    sparse_mask = np.array(bh_extract.sparse_mask)
    sel = np.logical_and(
        np.logical_and(sparse_mask[:, 1] >= 0, sparse_mask[:, 1] < 255),
        np.logical_and(sparse_mask[:, 0] >= 0, sparse_mask[:, 0] < 255),
    )

    flat_index = (sparse_mask[sel][:, 0]) * sinc_npx + (sparse_mask[sel][:, 1])
    sinc_integral = np.sum(sinc2d.ravel()[flat_index])
    assert np.abs(sinc_integral - np.nansum(extracted_data.ravel())) < 1.0e-8
    v = np.nanargmax(extracted_data)
    vx = v % extracted_data.shape[3]
    vy = v // extracted_data.shape[3]
    cextracted = (
        extracted_window_extents[0] + vx,
        extracted_window_extents[2] + vy,
    )
    v = np.nanargmax(sinc2d)
    sincvx = v % sinc_npx
    sincvy = v // sinc_npx
    csinc = tuple([sincvx, sincvy])
    assert csinc == cextracted

    # test case 4
    vals2 = np.array([[-20, -120], [0, 60], [40, -60]])
    vals3 = np.array([[-20, 58], [-40, 80], [20, 100]])
    bh2 = BoundingConvexHull(vals2)
    bh3 = BoundingConvexHull(vals3)
    assert bh.overlaps_with(bh2)
    assert not bh.overlaps_with(bh3)
    assert not bh2.overlaps_with(bh3)

    # test case 5
    assert (-1000, -1000) not in bh
    assert (30, 0) not in bh
    assert (0, 0) not in bh
    assert (-40, 30) in bh

    # test case 6
    bb = BoundingBox(-14, 20, 30, 49)
    assert bb.centre == [3, 39]
    assert bb.box_npx == (35, 20)
    assert bb.mask.shape == bb.box_npx[::-1]
    assert bb.area == 35 * 20

    assert np.sum(bb.mask) == bb.area
    assert (-15, 35) not in bb
    assert (0, 35) in bb

    bb2 = BoundingBoxFactory.AxisAlignedBoundingBox(bb)  # enforce odd
    assert bb2.box_npx == (35, 21)
    assert bb2.area == 35 * 21
    assert (bb.sparse_mask == bb2.sparse_mask).all()
    assert (-15, 35) not in bb2
    assert (0, 35) in bb2

    bb3 = BoundingBoxFactory.AxisAlignedBoundingBox(bb, square=True)  # enforce odd
    assert bb3.box_npx[0] == bb3.box_npx[1]
    assert bb3.box_npx[0] % 2 == 1  # enforce odd
    assert bb3.area == bb3.box_npx[0] ** 2
    assert (bb.sparse_mask == bb3.sparse_mask).all()
    assert (-15, 35) not in bb2
    assert (0, 35) in bb2

    # test case 7
    bb4s = BoundingBoxFactory.SplitBox(bb, nsubboxes=3)
    assert len(bb4s) == 9
    xlims = [(np.min(c.corners[:, 0]), np.max(c.corners[:, 0])) for c in bb4s][0:3]
    ylims = [(np.min(c.corners[:, 1]), np.max(c.corners[:, 1])) for c in bb4s][0::3]
    assert np.all(xlims == np.array([(-14, -3), (-2, 9), (10, 20)]))
    assert np.all(ylims == np.array([(30, 36), (37, 43), (44, 49)]))
    assert np.sum([b.area for b in bb4s]) == bb.area

    for bb4 in bb4s:
        assert bb4.area == np.sum(bb4.mask)

    # test case 8
    bb5 = BoundingBox(-14, 20, 30, 50)
    assert bb5.box_npx == (35, 21)
    bb6 = BoundingBoxFactory.PadBox(bb5, 41, 27)
    assert bb6.box_npx == (41, 27)
    assert bb5.centre == bb6.centre
    assert np.sum(bb5.mask) == np.sum(bb6.mask)
    bb7s = list(map(lambda x: BoundingBoxFactory.PadBox(x, 17, 11), bb4s))
    assert all([b.box_npx == (17, 11) for b in bb7s])
    assert np.sum([np.sum(b.mask) for b in bb7s]) == np.sum(
        [np.sum(b.mask) for b in bb4s]
    )

    # test case 9
    facet_regions = list(
        map(
            lambda f: BoundingBoxFactory.PadBox(f, 63, 63),
            BoundingBoxFactory.SplitBox(
                BoundingBoxFactory.AxisAlignedBoundingBox(bh_extract),
                nsubboxes=5,
            ),
        )
    )
    facets = list(
        map(
            lambda pf: BoundingConvexHull.regional_data(pf, sinc2d, oob_value=np.nan),
            facet_regions,
        )
    )
    stitched_image, stitched_region = BoundingBox.project_regions(
        [f[0] for f in facets], facet_regions
    )
    assert np.abs(sinc_integral - np.nansum([np.nansum(f[0]) for f in facets])) < 1.0e-8
    assert np.abs(sinc_integral - np.sum(stitched_image)) < 1.0e-8
    v = np.argmax(stitched_image)
    vx = v % stitched_image.shape[3]
    vy = v // stitched_image.shape[3]
    cstitched = (
        np.min(stitched_region.corners[:, 0]) + vx,
        np.min(stitched_region.corners[:, 1]) + vy,
    )
    assert cstitched == csinc

    # test case 10
    olap_box1 = BoundingBox(110, 138, 110, 135)
    olap_box2 = BoundingBox(115, 150, 109, 150)
    olap_box3 = BoundingBox(125, 130, 125, 130)
    BoundingConvexHull.normalize_masks([olap_box1, olap_box2, olap_box3])
    ext1 = BoundingConvexHull.regional_data(olap_box1, sinc2d)[0]
    ext2 = BoundingConvexHull.regional_data(olap_box2, sinc2d)[0]
    ext3 = BoundingConvexHull.regional_data(olap_box3, sinc2d)[0]
    olaps_stitched_image, olaps_stitched_region = BoundingBox.project_regions(
        [ext1, ext2, ext3], [olap_box1, olap_box2, olap_box3]
    )
    v = np.nanargmax(olaps_stitched_image)
    vx = v % olaps_stitched_image.shape[3]
    vy = v // olaps_stitched_image.shape[3]
    cstitched_olap = (
        np.min(olaps_stitched_region.corners[:, 0]) + vx,
        np.min(olaps_stitched_region.corners[:, 1]) + vy,
    )
    assert cstitched_olap == csinc
    assert np.abs(1.0 - np.nanmax(olaps_stitched_image)) < 1.0e-8

    # visual inspection
    if debug:
        from matplotlib import pyplot as plt

        plt.figure(figsize=(7, 2.5))
        plt.title("Winding, normals and masking check")
        for h in [bh, bh2, bh3]:
            for ei, e in enumerate(h.edges):
                plt.plot(e[:, 0], e[:, 1], "r--")
                plt.text(e[0, 0], e[0, 1], str(ei))

        plt.plot(bh.edge_midpoints[:, 0], bh.edge_midpoints[:, 1], "ko")
        for e, n in zip(bh.edge_midpoints, normalized_normals):
            p0 = e
            p = e + n * 6
            plt.plot([p0[0], p[0]], [p0[1], p[1]], "b--", lw=2)

        plt.scatter(vals[:, 0], vals[:, 1])
        plt.imshow(
            mask,
            extent=[
                np.min(vals[:, 0]),
                np.max(vals[:, 0]),
                np.max(vals[:, 1]),
                np.min(vals[:, 1]),
            ],
        )

        plt.grid(True)
        plt.savefig("/tmp/winding.png")

        plt.figure(figsize=(7, 2.5))
        plt.title("Data extraction check (global)")
        for h in [bh_extract]:
            for ei, e in enumerate(h.edges):
                plt.plot(e[:, 0], e[:, 1], "r--")
        plt.imshow(sinc2d[0, 0, :, :], extent=[0, sinc_npx, sinc_npx, 0])
        plt.grid(True)
        plt.savefig("/tmp/extract_global.png")

        plt.figure(figsize=(7, 2.5))
        plt.title("Data extraction check (local)")
        for h in [bh_extract]:
            for ei, e in enumerate(h.edges):
                plt.plot(e[:, 0], e[:, 1], "r--")
        plt.imshow(
            extracted_data[0, 0, :, :],
            extent=[
                extracted_window_extents[0],
                extracted_window_extents[1],
                extracted_window_extents[3],
                extracted_window_extents[2],
            ],
        )
        plt.savefig("/tmp/extract_local.png")

        plt.figure(figsize=(7, 2.5))
        plt.title("Faceting check")
        for h in [bh_extract]:
            for ei, e in enumerate(h.edges):
                plt.plot(e[:, 0], e[:, 1], "r--")
        for f in facet_regions:
            for ei, e in enumerate(f.edges):
                plt.plot(e[:, 0], e[:, 1], "co--")

        plt.imshow(
            stitched_image[0, 0, :, :],
            extent=[
                np.min(stitched_region.corners[:, 0]),
                np.max(stitched_region.corners[:, 0]),
                np.max(stitched_region.corners[:, 1]),
                np.min(stitched_region.corners[:, 1]),
            ],
        )
        plt.savefig("/tmp/facet.png")

        plt.figure(figsize=(7, 2.5))
        plt.title("Overlapping faceting check")
        for f in [olap_box1, olap_box2, olap_box3]:
            for ei, e in enumerate(f.edges):
                plt.plot(e[:, 0], e[:, 1], "co--")

        plt.imshow(
            olaps_stitched_image[0, 0, :, :],
            extent=[
                np.min(olaps_stitched_region.corners[:, 0]),
                np.max(olaps_stitched_region.corners[:, 0]),
                np.max(olaps_stitched_region.corners[:, 1]),
                np.min(olaps_stitched_region.corners[:, 1]),
            ],
        )
        plt.xlim(
            (
                np.min(olaps_stitched_region.corners[:, 0]) - 15,
                np.max(olaps_stitched_region.corners[:, 0]) + 15,
            )
        )
        plt.ylim(
            (
                np.min(olaps_stitched_region.corners[:, 1]) - 15,
                np.max(olaps_stitched_region.corners[:, 1]) + 15,
            )
        )
        plt.savefig("/tmp/overlap_facet.png")


if __name__ == "__main__":
    test_hull_construction()
