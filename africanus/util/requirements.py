# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .rtd import on_rtd

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
_have_packages = set([p for p in _check_packages if _package_exists(p)])

def have_packages(*args):
    unknown_requirements = tuple(p for p in args if p not in _check_packages)

    if len(unknown_requirements) > 0:
        raise ValueError("The following requirements are not registered: %s"
                            % unknown_requirements)

    return all(p in _have_packages for p in args)

class MissingPackageException(Exception):
    def __init__(self, *packages):
        super(MissingPackageException, self).__init__(
            "The following packages must be installed: %s" % packages)
