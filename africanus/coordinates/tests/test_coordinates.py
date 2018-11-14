from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from africanus.coordinates import (radec_to_lmn as np_radec_to_lmn,
                                   lmn_to_radec as np_lmn_to_radec)


def test_radec_to_lmn():
    """ Tests that basics run """
    radec = np.random.random((10, 2))*np.pi
    phase_centre = np.random.random(2)*np.pi

    lmn = np_radec_to_lmn(radec, phase_centre)
    final_radec = np_lmn_to_radec(lmn, phase_centre)

    # lmn = np_radec_to_lmn(radec)
    # final_radec = np_lmn_to_radec(lmn)


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
                                            lmn_to_radec as da_lmn_to_radec)

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
    da_radec = da_lmn_to_radec(da_lmn, da_phase_centre)

    assert np.all(da_lmn.compute() == np_lmn)
    assert np.all(da_radec.compute() == np_radec)
