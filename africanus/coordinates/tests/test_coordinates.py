from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from africanus.coordinates import (radec_to_lmn as np_radec_to_lmn,
                                   radec_to_lm as np_radec_to_lm,
                                   lmn_to_radec as np_lmn_to_radec,
                                   lm_to_radec as np_lm_to_radec)


def test_radec_to_lmn():
    """ Tests that basics run """
    radec = np.random.random((10, 2))*np.pi
    phase_centre = np.random.random(2)*np.pi

    lmn = np_radec_to_lmn(radec, phase_centre)
    final_radec = np_lmn_to_radec(lmn, phase_centre)

    lm = lmn[:, :2]

    assert np.all(np_radec_to_lm(radec, phase_centre) == lm)
    assert np.all(np_lm_to_radec(lm, phase_centre) == final_radec)

    zpc = np.zeros((2,), dtype=radec.dtype)

    # Test missing phase centre cases
    assert np.all(np_radec_to_lmn(radec) == np_radec_to_lmn(radec, zpc))
    assert np.all(np_radec_to_lm(radec) == np_radec_to_lm(radec, zpc))
    assert np.all(np_lmn_to_radec(lmn) == np_lmn_to_radec(lmn, zpc))
    assert np.all(np_lm_to_radec(lm) == np_lm_to_radec(lm, zpc))


@pytest.mark.xfail
def test_radec_to_lmn_wraps():
    """ Test that the radec can be recovered exactly """
    radec = np.random.random((10, 2))*np.pi
    phase_centre = np.random.random(2)*np.pi

    lmn = np_radec_to_lmn(radec, phase_centre)
    final_radec = np_lmn_to_radec(lmn, phase_centre)
    final_radec = (final_radec + np.pi) % (2 * np.pi) - np.pi

    assert np.allclose(final_radec, radec)


def test_dask_radec_to_lmn():
    """ Test that dask version matches numpy version """
    da = pytest.importorskip("dask.array")

    from africanus.coordinates.dask import (radec_to_lmn as da_radec_to_lmn,
                                            radec_to_lm as da_radec_to_lm,
                                            lmn_to_radec as da_lmn_to_radec,
                                            lm_to_radec as da_lm_to_radec)

    source_chunks = (5, 5, 5)
    coord_chunks = (2,)

    source = sum(source_chunks)
    coords = sum(coord_chunks)

    radec = np.random.random((source, coords))*10
    da_radec = da.from_array(radec, chunks=(source_chunks, coord_chunks))

    phase_centre = np.random.random(coord_chunks)*np.pi
    da_phase_centre = da.from_array(phase_centre, chunks=(coord_chunks,))

    np_lmn = np_radec_to_lmn(radec, phase_centre)
    np_radec = np_lmn_to_radec(np_lmn, phase_centre)

    da_lmn = da_radec_to_lmn(da_radec, da_phase_centre)
    radec_result = da_lmn_to_radec(da_lmn, da_phase_centre)

    da_lm = da_radec_to_lm(da_radec, da_phase_centre)
    radec_result_2 = da_lm_to_radec(da_lm, da_phase_centre)

    assert np.all(da_lmn.compute() == np_lmn)
    assert np.all(radec_result.compute() == np_radec)
    assert np.all(da_lm.compute() == np_lmn[:, :2])
    # Going to radec via lmn and lm makes no difference
    assert da.all(radec_result == radec_result_2).compute()

    # Going to lm/lmn makes from radec is equivalent
    da_lm = da.from_array(np_lmn[:, :2], chunks=(source_chunks, coord_chunks))
    result = da_radec_to_lm(da_radec, da_phase_centre).compute()
    assert np.all(result == np_lmn[:, :2])
    v1 = da_lm_to_radec(da_lm, da_phase_centre).compute()
    v2 = radec_result.compute()
    assert np.allclose(v1, v2)

    # Test missing phase centre cases
    zpc = da.zeros((2,), dtype=radec.dtype, chunks=(2,))
    assert da.all(da_radec_to_lmn(da_radec) == da_radec_to_lmn(da_radec, zpc)).compute()  # noqa
    assert da.all(da_radec_to_lm(da_radec) == da_radec_to_lm(da_radec, zpc)).compute()    # noqa
    assert da.all(da_lmn_to_radec(da_lmn) == da_lmn_to_radec(da_lmn, zpc)).compute()      # noqa
    assert da.all(da_lm_to_radec(da_lm) == da_lm_to_radec(da_lm, zpc)).compute()          # noqa
