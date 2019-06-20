# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from africanus.io.ms.creation import empty_ms

import numpy as np
import pytest


@pytest.mark.parametrize("nrow", [10])
@pytest.mark.parametrize("nchan", [16])
@pytest.mark.parametrize("ncorr", [1, 4])
@pytest.mark.parametrize("add_imaging_cols", [True, False])
def test_empty_ms(nrow, nchan, ncorr, add_imaging_cols, tmpdir):
    pt = pytest.importorskip('pyrap.tables')

    ms = str(tmpdir) + os.pathsep + "test.ms"
    empty_ms(ms, nchan, ncorr, add_imaging_cols=add_imaging_cols)
    imaging_cols = set(["MODEL_DATA", "CORRECTED_DATA", "IMAGING_WEIGHT"])

    with pt.table(ms, readonly=False, ack=False) as T:
        # Add rows, get data, put data
        T.addrows(nrow)
        data = T.getcol("DATA")
        assert data.shape == (nrow, nchan, ncorr)
        T.putcol("DATA", np.zeros_like(data))

        table_cols = set(T.colnames())

        if imaging_cols is True:
            assert T.getcol("MODEL_DATA").shape == (nrow, nchan, ncorr)
            assert imaging_cols in table_cols
        else:
            assert imaging_cols not in table_cols
