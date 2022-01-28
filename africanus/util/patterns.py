from collections import OrderedDict
from inspect import getattr_static
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


class LazyProxy:
    """

    """

    __slots__ = (
        "__lazy_fn__",
        "__lazy_finaliser__",
        "__lazy_args__",
        "__lazy_kwargs__",
        "__lazy_object__",
        "__lazy_lock__",
        "__weakref__"
    )

    SLOTS = set(__slots__)

    def __init__(self, fn, *args, **kwargs):
        ex = ValueError("fn must be a callable or a tuple of two callables")

        if isinstance(fn, tuple):
            if len(fn) != 2:
                raise ex

            if not callable(fn[0]):
                raise ex

            if fn[1] is not None and not callable(fn[1]):
                raise ex

            self.__lazy_fn__, self.__lazy_finaliser__ = fn
        elif callable(fn):
            self.__lazy_fn__, self.__lazy_finaliser__ = fn, None
        else:
            raise ex

        self.__lazy_args__ = args
        self.__lazy_kwargs__ = kwargs
        self.__lazy_lock__ = Lock()

    def __eq__(self, other):
        return (
            isinstance(other, LazyProxy) and
            self.__lazy_fn__ == other.__lazy_fn__ and
            self.__lazy_finaliser__ == other.__lazy_finaliser__ and
            self.__lazy_args__ == other.__lazy_args__ and
            self.__lazy_kwargs__ == other.__lazy_kwargs__)

    @classmethod
    def from_args(cls, fn, args, kwargs):
        return cls(fn, *args, **kwargs)

    def __reduce__(self):
        return (self.from_args,
                (((self.__lazy_fn__, self.__lazy_finaliser__)
                 if self.__lazy_finaliser__ else self.__lazy_fn__),
                 self.__lazy_args__, self.__lazy_kwargs__))

    def __getattr__(self, name):
        if name == "__lazy_object__":
            # The __lazy_object__ has not been created at this point,
            # acquire the creation lock
            with self.__lazy_lock__:
                # getattr_static returns a descriptor for __slots__
                descriptor = getattr_static(self, "__lazy_object__")

                try:
                    # __lazy_object__ may have been created prior to
                    # lock acquisition, try to return it
                    return descriptor.__get__(self)
                except AttributeError:
                    # Create __lazy_object__
                    lazy_obj = self.__lazy_fn__(*self.__lazy_args__,
                                                **self.__lazy_kwargs__)
                    self.__lazy_object__ = lazy_obj

                    # Create finaliser if provided
                    if self.__lazy_finaliser__:
                        weakref.finalize(self, self.__lazy_finaliser__,
                                         lazy_obj)

                return lazy_obj

        # Proxy attribute on the __lazy_object__
        obj = self if name in self.SLOTS else self.__lazy_object__
        return object.__getattribute__(obj, name)

    def __setattr__(self, name, value):
        obj = self if name in self.SLOTS else self.__lazy_object__
        return object.__setattr__(obj, name, value)

    def __delattr__(self, name):
        if name in self.SLOTS:
            raise ValueError(f"{name} may not be deleted")

        return object.__delattr__(self.__lazy_object__, name)


class LazyProxyMultiton(LazyProxy, metaclass=Multiton):
    pass
