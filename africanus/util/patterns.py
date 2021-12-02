from collections import OrderedDict
from threading import Lock
from warnings import warn
import weakref

from numpy import ndarray


def freeze(arg):
    """ Recursively generates a hashable object from arg """
    if isinstance(arg, set):
        return tuple(map(freeze, sorted(arg)))
    elif isinstance(arg, (tuple, list)):
        return tuple(map(freeze, arg))
    elif isinstance(arg, (dict, OrderedDict)):
        return frozenset((freeze(k), freeze(v)) for k, v
                         in sorted(arg.items()))
    elif isinstance(arg, ndarray):
        if arg.nbytes > 10:
            warn(f"freezing ndarray of size {arg.nbytes} "
                 f" is probably inefficient")
        return freeze(arg.tolist())
    else:
        return arg


class Multiton(type):
    """General Multiton metaclass

    Implementation of the `Multiton`_ pattern which controls the creation
    of instances of a class.

    ```python
    class A(metaclass=Multiton):
        def __init__(self, *args, **kw):
            self.args = args
            self.kw = kw

    assert A(1) is A(1)
    assert A(1, "bob") is not A(1)

    .. _Multiton: https://en.wikipedia.org/wiki/Multiton_pattern

    """
    MISSING = object()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__cache = weakref.WeakValueDictionary()
        self.__lock = Lock()

    def __call__(self, *args, **kwargs):
        key = freeze(args + (kwargs if kwargs else self.MISSING,))

        # Double-checked locking
        # https://en.wikipedia.org/wiki/Double-checked_locking
        try:
            return self.__cache[key]
        except KeyError:
            with self.__lock:
                try:
                    return self.__cache[key]
                except KeyError:
                    instance = type.__call__(self, *args, **kwargs)
                    self.__cache[key] = instance
                    return instance
