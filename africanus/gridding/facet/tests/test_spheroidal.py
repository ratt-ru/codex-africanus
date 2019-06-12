# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from africanus.gridding.facet.spheroidal import (delta_n_coefficients,
                                                 np_spheroidal_2d,
                                                 spheroidal_2d,
                                                 spheroidal_aa_filter as spaaf,
                                                 wplanes)

@pytest.fixture
def freqs():
    return np.linspace(.856e9, 2*.856e9, 4)

@pytest.mark.parametrize("support", [11])
@pytest.mark.parametrize("spheroidal_support", [111])
@pytest.mark.parametrize("npix", [1025])
@pytest.mark.parametrize("wlayers", [7])
@pytest.mark.parametrize("maxw", [30000])
@pytest.mark.parametrize("cell_size", [1.3])
@pytest.mark.parametrize("oversampling", [11])
@pytest.mark.parametrize("lm_shift", [(1e-8, 1e-8)])
def test_spheroidal_vs_ddfacet(support, spheroidal_support,
                               npix, wlayers, maxw,
                               cell_size, oversampling,
                               lm_shift,
                               freqs, tmpdir):
    pytest.importorskip("DDFacet")

    from DDFacet.Imager.ModCF import SpheMachine, ClassWTermModified, Give_dn
    from DDFacet.Array.shared_dict import SharedDict
    from DDFacet.ToolsDir import ModTaper

    # Test the spheroidal's match
    numba_spheroidal = spheroidal_2d(support)
    numpy_spheroidal = np_spheroidal_2d(support)
    ddf_spheroidal = ModTaper.Sphe2D(support)

    # Numpy and Numba implementations match
    assert_array_almost_equal(numba_spheroidal, numpy_spheroidal)

    # Africanus and DDFacet implementations match
    assert_array_almost_equal(numba_spheroidal, ddf_spheroidal)

    # Test that SpheMachine and spheroidal_aa_filter do the same thing
    cf, fcf, ifzfcf = spaaf(npix, support=support,
                            spheroidal_support=spheroidal_support)

    sm = SpheMachine(Support=support, SupportSpheCalc=spheroidal_support)
    ddf_cf, ddf_fcf, ddf_ifzfcf = sm.MakeSphe(npix)

    assert_array_almost_equal(cf, ddf_cf)
    assert_array_almost_equal(fcf, ddf_fcf)
    assert_array_almost_equal(ifzfcf, ddf_ifzfcf)

    radius_deg = (npix / 2.0) * cell_size / 3600.
    radius_lm = radius_deg * np.pi / 180.

    # Check that we produce the same delta n
    ddf_cl, ddf_cm, ddf_dn = Give_dn(*lm_shift, rad=radius_lm)
    afr_cl, afr_cm, afr_dn = delta_n_coefficients(*lm_shift, radius=radius_lm)
    assert_array_almost_equal(ddf_cl, afr_cl)
    assert_array_almost_equal(ddf_cm, afr_cm)
    assert_array_almost_equal(ddf_dn, afr_dn)

    # Now create a full on DDFacet W-term class
    cf_dict = SharedDict(str(tmpdir))

    wterm = ClassWTermModified(cf_dict=cf_dict, OverS=oversampling,
                               Sup=support, Npix=npix,  Cell=cell_size,
                               Freqs=freqs, Nw=wlayers, wmax=maxw,
                               lmShift=lm_shift)

    # Check that spheroidals match again
    ddf_cf, ddf_fcf, ddf_ifzfcf = wterm.SpheM.MakeSphe(npix)
    cf, fcf, ifzfcf = spaaf(npix, support=support)

    assert_array_almost_equal(cf, ddf_cf)
    assert_array_almost_equal(fcf, ddf_fcf)
    assert_array_almost_equal(ifzfcf, ddf_ifzfcf)

    # Create codex-africanus wplanes
    wcf, wcf_conj = wplanes(wlayers, cell_size, support, maxw,
                            npix, oversampling,
                            lm_shift, freqs)

    # Same number of wplanes
    assert len(wcf) == len(wterm.Wplanes) == wlayers
    assert len(wcf_conj) == len(wterm.WplanesConj) == wlayers

    # Check that codex and DDFacet wplanes match
    for w, (cf, cf_conj) in enumerate(zip(wcf, wcf_conj)):
        assert_array_almost_equal(cf, wterm.Wplanes[w])
        assert_array_almost_equal(cf_conj, wterm.WplanesConj[w])
