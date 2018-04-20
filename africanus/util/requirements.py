# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from .docs import on_rtd

log = logging.getLogger(__name__)

if on_rtd():
    # Some dependencies are mock objects within readthedocs
    # which causes importlib to fail
    def _package_exists(package):
        return True
else:
    import importlib

    def _package_exists(package):
        try:
            importlib.import_module(package)
            return True
        except ImportError:
            return False

_check_packages = set(['dask.array', 'toolz'])


def have_packages(*args):
    unknown_requirements = tuple(p for p in args if p not in _check_packages)

    if len(unknown_requirements) > 0:
        log.debug("The following requirements "
                  "are not registered: %s", (unknown_requirements,))

    return all(_package_exists(p) for p in args)


class MissingPackageException(Exception):
    def __init__(self, *packages):
        super(MissingPackageException, self).__init__(
            "The following packages must be installed: %s" % (packages,))
