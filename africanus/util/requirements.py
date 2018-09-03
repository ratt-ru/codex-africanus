# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import logging
import sys

from decorator import decorate

from .docs import on_rtd

try:
    import pytest
except ImportError:
    pass


log = logging.getLogger(__name__)


class MissingPackageException(Exception):
    def __init__(self, *packages):
        super(MissingPackageException, self).__init__(
            "The following packages must be installed: %s" % (packages,))


def requires_optional(*requirements):
    """
    Decorator which returns either the original function,
    or a dummy function which raises a
    :class:`MissingPackageException` when called,
    depending on whether the supplied ``requirements``
    are present.

    If packages are missing and called within a test, the
    dummy function will call :func:`pytest.skip`.

    Used in the following way:

    .. code-block:: python

        try:
            from scipy import interpolate
        except ImportError:
            pass

        @requires_optional('scipy')
        def function(*args, **kwargs):
            return interpolate(...)


    Parameters
    ----------
    requirements : iterable of string
        Sequence of packages required by the decorated functions

    Returns
    -------
    callable
        Either the original function if all ``requirements``
        are available or a dummy function that throws
        a :class:`MissingPackageException` or skips within
        a pytest.
    """
    have_requirements = True
    missing_requirements = []

    # Try imports if we're not on RTD
    if not on_rtd():
        for package in requirements:
            try:
                importlib.import_module(package)
            except ImportError:
                missing_requirements.append(package)
                have_requirements = False
            else:
                pass

    def _function_decorator(fn):
        # Return a bare wrapper if we're on RTD
        if on_rtd():
            def _wrapper(f, *arg, **kwargs):
                """ Empty docstring """
                pass

            return decorate(fn, _wrapper)
        # We don't have requirements, produce
        # a failing wrapper
        elif not have_requirements:
            def _wrapper(f, *args, **kwargs):
                """ Empty docstring """
                if getattr(sys, "_called_from_test", False):
                    pytest.skip("Missing requirements %s" %
                                missing_requirements)
                else:
                    raise MissingPackageException(*missing_requirements)

            return decorate(fn, _wrapper)
        else:
            # Return the original function
            return fn

        return _wrapper

    return _function_decorator
