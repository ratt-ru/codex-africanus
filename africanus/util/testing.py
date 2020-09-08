# -*- coding: utf-8 -*-


try:
    from dask.utils import SerializableLock as Lock
except ImportError:
    from threading import Lock


__run_marker = {'in_pytest': False}
__run_marker_lock = Lock()


# Tag indicating that missing packages should generate an
# exception, regardless of the 'in_pytest' marker
# Used for testing exception raising behaviour
force_missing_pkg_exception = object()


def in_pytest():
    """ Return True if we're marked as executing inside pytest """
    with __run_marker_lock:
        return __run_marker['in_pytest']


def mark_in_pytest(in_pytest=True):
    """ Mark if we're in a pytest run """
    if type(in_pytest) is not bool:
        raise TypeError('in_pytest %s is not a boolean' % in_pytest)

    with __run_marker_lock:
        __run_marker['in_pytest'] = in_pytest
