#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys

import pytest

from africanus.util.requirements import (requires_optional,
                                         MissingPackageException)
from africanus.util.testing import force_missing_pkg_exception as force_tag


def test_requires_optional_missing_import():
    @requires_optional('sys', 'bob', force_tag)
    def f(*args, **kwargs):
        pass

    with pytest.raises(MissingPackageException) as e:
        f(1, a=2)

    assert ("f requires installation of the following packages: ('bob',)."
            in str(e.value))


def test_requires_optional_pass_import_error():
    assert 'clearly_missing_and_nonexistent_package' not in sys.modules

    try:
        import clearly_missing_and_nonexistent_package  # noqa
    except ImportError as e:
        me = e
    else:
        me = None

    with pytest.raises(ImportError) as e:
        @requires_optional('sys', 'os', me, force_tag)
        def f(*args, **kwargs):
            pass

    msg = str(e.value)
    assert "Successfully imported ['sys', 'os']" in msg
    assert "No module named" in msg
    assert "clearly_missing_and_nonexistent_package" in msg
