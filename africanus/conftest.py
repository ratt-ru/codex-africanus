# -*- coding: utf-8 -*-


from africanus.util.testing import mark_in_pytest


# content of conftest.py
def pytest_configure(config):
    mark_in_pytest(True)


def pytest_unconfigure(config):
    mark_in_pytest(False)
