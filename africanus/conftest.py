# -*- coding: utf-8 -*-

import importlib
import pytest

from africanus.util.testing import mark_in_pytest


@pytest.fixture
def cfg_parallel(request):
    """ Performs parallel configuration setting and module reloading """
    from africanus.config import config

    module, cfg = request.param

    assert isinstance(cfg, dict) and len(cfg) == 1

    # Get module object, because importlib.reload doesn't take strings
    mod = importlib.import_module(module)

    with config.set(cfg):
        importlib.reload(mod)

        cfg = cfg.copy().popitem()[1]

        if isinstance(cfg, dict):
            yield cfg['parallel']
        elif isinstance(cfg, bool):
            yield cfg
        else:
            raise TypeError("Unhandled cfg type %s" % type(cfg))

    importlib.reload(mod)



# content of conftest.py
def pytest_configure(config):
    mark_in_pytest(True)


def pytest_unconfigure(config):
    mark_in_pytest(False)
