from collections import OrderedDict
import inspect
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


INVALID_LAZY_CONTEXTS = set()

try:
    import dask.blockwise as db
except ImportError:
    pass
else:
    INVALID_LAZY_CONTEXTS.add(db.blockwise.__code__)

try:
    import dask.array as da
except ImportError:
    pass
else:
    INVALID_LAZY_CONTEXTS.add(da.blockwise.__code__)


class InvalidLazyContext(ValueError):
    pass


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

    __lazy_members__ = set(__slots__)

    def __init__(self, fn, *args, **kwargs):
        ex = ValueError("fn must be a callable or a tuple of two callables")

        if isinstance(fn, tuple):
            if (len(fn) != 2 or not callable(fn[0])
                    or (fn[1] and not callable(fn[1]))):
                raise ex

            self.__lazy_fn__, self.__lazy_finaliser__ = fn
        elif callable(fn):
            self.__lazy_fn__, self.__lazy_finaliser__ = fn, None
        else:
            raise ex

        self.__lazy_args__ = args
        self.__lazy_kwargs__ = kwargs
        self.__lazy_lock__ = Lock()

    def __lazy_eq__(self, other):
        return (
            isinstance(other, LazyProxy) and
            self.__lazy_fn__ == other.__lazy_fn__ and
            self.__lazy_finaliser__ == other.__lazy_finaliser__ and
            self.__lazy_args__ == other.__lazy_args__ and
            self.__lazy_kwargs__ == other.__lazy_kwargs__)

    def __lazy_hash__(self):
        return (
            self.__lazy_fn__,
            self.__lazy_finaliser__,
            self.__lazy_args__,
            frozenset(self.__lazy_kwargs__.items())
        ).__hash__()

    @classmethod
    def __lazy_from_args__(cls, fn, args, kwargs):
        return cls(fn, *args, **kwargs)

    @classmethod
    def __lazy_in_valid_frame__(cls, frame, depth=10):
        """
        Check that we're not trying to create the lazy object
        in an invalid call frame. We do this to prevent
        frameworks like dask inadvertently creating the
        LazyProxy via dunder method calls. Once the lazy object
        is created, this should no longer be called.

        Raises
        ------
        InvalidLazyContext:
            Raised if the call stack contains a call to a
            problematic function (like `dask.array.blockwise`)
        """
        while frame and depth > 0:
            if frame.f_code in INVALID_LAZY_CONTEXTS:
                raise InvalidLazyContext(
                    f"Attempted to create a LazyObject within a call "
                    f"to {frame.f_code.co_name}")

            depth -= 1
            frame = frame.f_back

    @classmethod
    def __lazy_obj_from_args__(cls, self):
        # Check that we're in a valid call frame
        cls.__lazy_in_valid_frame__(inspect.currentframe())

        # getattr_static returns a descriptor for __slots__
        descriptor = getattr_static(self, "__lazy_object__")

        # Double-locking pattern follows, perhaps __lazy_object__
        # has been created, in which case, return it
        try:
            return descriptor.__get__(self)
        except AttributeError:
            pass

        # Acquire the creation lock
        with self.__lazy_lock__:
            # getattr_static returns a descriptor for __slots__
            descriptor = getattr_static(self, "__lazy_object__")

            try:
                # __lazy_object__ may have been created prior to
                # lock acquisition, attempt to return it again
                return descriptor.__get__(self)
            except AttributeError:
                # Create __lazy_object__
                lazy_obj = self.__lazy_fn__(*self.__lazy_args__,
                                            **self.__lazy_kwargs__)
                self.__lazy_object__ = lazy_obj

                # Create finaliser if provided
                if self.__lazy_finaliser__:
                    weakref.finalize(self,
                                     self.__lazy_finaliser__,
                                     lazy_obj)

            return lazy_obj

    def __getattr__(self, name):
        if name == "__lazy_object__":
            return LazyProxy.__lazy_obj_from_args__(self)

        try:
            return object.__getattribute__(self.__lazy_object__, name)
        except InvalidLazyContext as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        obj = self if name in self.__lazy_members__ else self.__lazy_object__
        return object.__setattr__(obj, name, value)

    def __delattr__(self, name):
        if name in self.__lazy_members__:
            raise ValueError(f"{name} may not be deleted")

        return object.__delattr__(self.__lazy_object__, name)

    def __reduce__(self):
        return (self.__lazy_from_args__,
                (((self.__lazy_fn__, self.__lazy_finaliser__)
                 if self.__lazy_finaliser__ else self.__lazy_fn__),
                 self.__lazy_args__, self.__lazy_kwargs__))


class LazyProxyMultiton(LazyProxy, metaclass=Multiton):
    pass
