# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from decorator import decorate
from numba import types

from africanus.util.docs import on_rtd

if on_rtd():
    # Fake decorators when on readthedocs
    def generated_jit(*args, **kwargs):
        def decorator(fn):
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            return decorate(fn, wrapper)

        return decorator

    def jit(*args, **kwargs):
        def decorator(fn):
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            return decorate(fn, wrapper)

        return decorator
else:
    from numba import jit, generated_jit  # noqa


def is_numba_type_none(arg):
    """
    Returns True if the numba type represents None


    Parameters
    ----------
    arg : :class:`numba.Type`
        The numba type

    Returns
    -------
    boolean
        True if the type represents None
    """
    return (isinstance(arg, types.misc.NoneType) or
            (isinstance(arg, types.misc.Omitted) and arg.value is None))
