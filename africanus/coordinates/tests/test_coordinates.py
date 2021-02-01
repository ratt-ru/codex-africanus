
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest

from africanus.coordinates import (radec_to_lmn as np_radec_to_lmn,
                                   radec_to_lm as np_radec_to_lm,
                                   lmn_to_radec as np_lmn_to_radec,
                                   lm_to_radec as np_lm_to_radec)

from africanus.coordinates.coordinates import astropy_radec_to_lmn


def test_radec_to_lmn():
    """ Tests that basics run """

    np.random.seed(42)

    radec = np.random.random((10, 2))
    phase_centre = np.random.random(2)

    lmn = np_radec_to_lmn(radec, phase_centre)
    final_radec = np_lmn_to_radec(lmn, phase_centre)

    lm = lmn[:, :2]

    assert_array_equal(np_radec_to_lm(radec, phase_centre), lm)
    assert_array_equal(np_lm_to_radec(lm, phase_centre), final_radec)

    zpc = np.zeros((2,), dtype=radec.dtype)

    # Test missing phase centre cases
    assert_array_equal(np_radec_to_lmn(radec), np_radec_to_lmn(radec, zpc))
    assert_array_equal(np_radec_to_lm(radec), np_radec_to_lm(radec, zpc))
    assert_array_equal(np_lmn_to_radec(lmn), np_lmn_to_radec(lmn, zpc))
    assert_array_equal(np_lm_to_radec(lm), np_lm_to_radec(lm, zpc))


def test_radec_to_lmn_astropy():
    """ Check that our code agrees with astropy """

    np.random.seed(42)

    astropy = pytest.importorskip('astropy')
    SkyCoord = astropy.coordinates.SkyCoord
    units = astropy.units

    radec = np.random.random((10, 2))
    phase_centre = np.random.random(2)

    lmn = np_radec_to_lmn(radec, phase_centre)

    ast_radec = SkyCoord(radec[:, 0], radec[:, 1], unit=units.rad)
    ast_phase_centre = SkyCoord(phase_centre[0], phase_centre[1],
                                unit=units.rad)
    ast_lmn = astropy_radec_to_lmn(ast_radec, ast_phase_centre)

    assert_array_almost_equal(ast_lmn, lmn)


def test_radec_to_lmn_wraps():
    """ Test that the radec can be recovered exactly """

    np.random.seed(42)

    radec = np.random.random((10, 2))
    phase_centre = np.random.random(2)

    lmn = np_radec_to_lmn(radec, phase_centre)
    final_radec = np_lmn_to_radec(lmn, phase_centre)
    final_radec = (final_radec + np.pi) % (2 * np.pi) - np.pi

    assert_array_almost_equal(final_radec, radec)


def test_dask_radec_to_lmn():
    """ Test that dask version matches numpy version """
    da = pytest.importorskip("dask.array")

    from africanus.coordinates.dask import (radec_to_lmn as da_radec_to_lmn,
                                            radec_to_lm as da_radec_to_lm,
                                            lmn_to_radec as da_lmn_to_radec,
                                            lm_to_radec as da_lm_to_radec)

    np.random.seed(42)

    source_chunks = (5, 5, 5)
    coord_chunks = (2,)

    source = sum(source_chunks)
    coords = sum(coord_chunks)

    radec = np.random.random((source, coords))*10
    da_radec = da.from_array(radec, chunks=(source_chunks, coord_chunks))

    phase_centre = np.random.random(coord_chunks)
    da_phase_centre = da.from_array(phase_centre, chunks=(coord_chunks,))

    np_lmn = np_radec_to_lmn(radec, phase_centre)
    np_radec = np_lmn_to_radec(np_lmn, phase_centre)

    da_lmn = da_radec_to_lmn(da_radec, da_phase_centre)
    radec_result = da_lmn_to_radec(da_lmn, da_phase_centre)

    da_lm = da_radec_to_lm(da_radec, da_phase_centre)
    radec_result_2 = da_lm_to_radec(da_lm, da_phase_centre)

    assert_array_equal(da_lmn.compute(), np_lmn)
    assert_array_equal(radec_result.compute(), np_radec)
    assert_array_equal(da_lm.compute(), np_lmn[:, :2])
    # Going to radec via lmn and lm makes no difference
    assert_array_equal(radec_result.compute(), radec_result_2.compute())

    # Going to lm/lmn makes from radec is equivalent
    da_lm = da.from_array(np_lmn[:, :2], chunks=(source_chunks, coord_chunks))
    result = da_radec_to_lm(da_radec, da_phase_centre).compute()
    assert_array_equal(result, np_lmn[:, :2])
    v1 = da_lm_to_radec(da_lm, da_phase_centre).compute()
    v2 = radec_result.compute()
    assert_array_equal(v1, v2)

    # Test missing phase centre cases
    zpc = da.zeros((2,), dtype=radec.dtype, chunks=(2,))

    assert_array_equal(da_radec_to_lmn(da_radec), da_radec_to_lmn(da_radec, zpc))  # noqa
    assert_array_equal(da_radec_to_lm(da_radec), da_radec_to_lm(da_radec, zpc))    # noqa
    assert_array_equal(da_lmn_to_radec(da_lmn), da_lmn_to_radec(da_lmn, zpc))      # noqa
    assert_array_equal(da_lm_to_radec(da_lm), da_lm_to_radec(da_lm, zpc))          # noqa
