#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

import importlib

import numpy as np
import pytest


@pytest.fixture
def cfg_rime_parallel(request):
    """ Performs parallel configuration setting and module reloading """
    from africanus.config import config

    module, cfg = request.param

    assert isinstance(cfg, dict) and len(cfg) == 1

    # Get module object, because importlib.reload doesn't take strings
    mod = importlib.import_module(module)

    with config.set(cfg):
        importlib.reload(mod)

        yield cfg.copy().popitem()[1]

    importlib.reload(mod)


@pytest.fixture
def wsrt_ants():
    """ Westerbork antenna positions """
    return np.array([
           [3828763.10544699,   442449.10566454,  5064923.00777],
           [3828746.54957258,   442592.13950824,  5064923.00792],
           [3828729.99081359,   442735.17696417,  5064923.00829],
           [3828713.43109885,   442878.2118934,  5064923.00436],
           [3828696.86994428,   443021.24917264,  5064923.00397],
           [3828680.31391933,   443164.28596862,  5064923.00035],
           [3828663.75159173,   443307.32138056,  5064923.00204],
           [3828647.19342757,   443450.35604638,  5064923.0023],
           [3828630.63486201,   443593.39226634,  5064922.99755],
           [3828614.07606798,   443736.42941621,  5064923.],
           [3828609.94224429,   443772.19450029,  5064922.99868],
           [3828601.66208572,   443843.71178407,  5064922.99963],
           [3828460.92418735,   445059.52053929,  5064922.99071],
           [3828452.64716351,   445131.03744105,  5064922.98793]],
        dtype=np.float64)
