# -*- coding: utf-8 -*-


from decorator import decorate
from numba import types

from africanus.util.docs import on_rtd

if on_rtd():
    # Fake decorators when on readthedocs
    def _fake_decorator(*args, **kwargs):
        def decorator(fn):
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            return decorate(fn, wrapper)

        return decorator

    cfunc = _fake_decorator
    generated_jit = _fake_decorator
    jit = _fake_decorator
    njit = _fake_decorator
    stencil = _fake_decorator
    overload = _fake_decorator
    register_jitable = _fake_decorator

else:
    from numba import cfunc, jit, njit, generated_jit, stencil  # noqa
    from numba.extending import overload, register_jitable  # noqa


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
