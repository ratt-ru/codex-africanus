# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from numpy.testing import assert_array_almost_equal
import pytest

from africanus.gridding.facet.spheroidal import (np_spheroidal_2d,
                                                 spheroidal_2d,
                                                 spheroidal_aa_filter as spaaf)



@pytest.mark.parametrize("support", [10, 11, 12])
@pytest.mark.parametrize("spheroidal_support", [110, 111, 112])
@pytest.mark.parametrize("npix", [1025])
def test_spheroidal_vs_ddfacet(support, spheroidal_support, npix, tmpdir):
    pytest.importorskip("DDFacet")

    from DDFacet.Imager.ModCF import SpheMachine
    from DDFacet.ToolsDir import ModTaper

    numba_spheroidal = spheroidal_2d(support)
    numpy_spheroidal = np_spheroidal_2d(support)
    ddf_spheroidal = ModTaper.Sphe2D(support)

    # Numpy and Numba implementations match
    assert_array_almost_equal(numba_spheroidal, numpy_spheroidal)

    # Africanus and DDFacet implementations match
    assert_array_almost_equal(numba_spheroidal, ddf_spheroidal)

    cf, fcf, ifzfcf = spaaf(npix, support=support,
                            spheroidal_support=spheroidal_support)

    sm = SpheMachine(Support=support, SupportSpheCalc=spheroidal_support)
    ddf_cf, ddf_fcf, ddf_ifzfcf = sm.MakeSphe(npix)

    assert_array_almost_equal(cf, ddf_cf)
    assert_array_almost_equal(fcf, ddf_fcf)
    assert_array_almost_equal(ifzfcf, ddf_ifzfcf)

    from DDFacet.Imager.ModCF import ClassWTermModified
    from DDFacet.Array.shared_dict import SharedDict

    cf_dict = SharedDict(path=str(tmpdir))
    wterm = ClassWTermModified(cf_dict=cf_dict, Sup=support, Npix=npix)

    ddf_cf, ddf_fcf, ddf_ifzfcf = wterm.SpheM.MakeSphe(npix)
    cf, fcf, ifzfcf = spaaf(npix, support=support)

    assert_array_almost_equal(cf, ddf_cf)
    assert_array_almost_equal(fcf, ddf_fcf)
    assert_array_almost_equal(ifzfcf, ddf_ifzfcf)

