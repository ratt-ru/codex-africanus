# -*- coding: utf-8 -*-

import os

from numba.core.runtime import rtsys
import pytest

from africanus.util.testing import mark_in_pytest


NUMBA_NRT_STATS_ENABLED = (
    os.environ.get("NUMBA_NRT_STATS", "false").lower() in {"true", "1"})

@pytest.fixture(scope="function", autouse=NUMBA_NRT_STATS_ENABLED)
def check_allocations():
    """ Check allocations match frees """
    try:
        yield
        start = rtsys.get_allocation_stats()
    finally:
        end = rtsys.get_allocation_stats()
        assert start.alloc - end.alloc == start.free - end.free


# content of conftest.py
def pytest_configure(config):
    mark_in_pytest(True)


def pytest_unconfigure(config):
    mark_in_pytest(False)
