"""Keep this file in sync with the dask-ms version"""

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

    Implementation of the `Multiton`_ pattern, which always returns a
    unique object for a unique set of arguments provided to a class
    constructor. For example, in the following, only a single instance
    of `A` with argument `1` is ever created.

    .. code-block:: python

        class A(metaclass=Multiton):
            def __init__(self, *args, **kw):
                self.args = args
                self.kw = kw

        assert A(1) is A(1)
        assert A(1, "bob") is not A(1)

    This is useful for ensuring that only a single instance of a
    heavy-weight resource such as files, sockets, thread/process pools
    or database connections is created in a single process,
    for a unique set of arguments.

    .. _Multiton: https://en.wikipedia.org/wiki/Multiton_pattern

    Notes
    -----
    Instantiation of object instances is thread-safe.

    """
    MISSING = object()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__cache = weakref.WeakValueDictionary()
        self.__lock = Lock()

    def __call__(cls, *args, **kwargs):
        signature = inspect.signature(cls.__init__)
        positional_in_kwargs = [p.name for p in signature.parameters.values()
                                if p.kind == p.POSITIONAL_OR_KEYWORD
                                and p.default == p.empty
                                and p.name in kwargs]

        if positional_in_kwargs:
            warn(f"Positional arguments {positional_in_kwargs} were "
                 f"supplied as keyword arguments to "
                 f"{cls.__init__}{signature}. "
                 f"This may create separate Multiton instances "
                 f"for what is intended to be a unique set of "
                 f"arguments.")

        key = freeze(args + (kwargs if kwargs else Multiton.MISSING,))

        # Double-checked locking
        # https://en.wikipedia.org/wiki/Double-checked_locking
        try:
            return cls.__cache[key]
        except KeyError:
            pass

        with cls.__lock:
            try:
                return cls.__cache[key]
            except KeyError:
                instance = type.__call__(cls, *args, **kwargs)
                cls.__cache[key] = instance
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
    """Lazy instantiation of a proxied object.

    A LazyProxy proxies an object which is lazily instantiated on first use.
    It is primarily useful for embedding references to heavy-weight resources
    in a dask graph, so they can be pickled and sent to other workers
    without immediately instantiating those resources.

    To this end, the proxy takes as arguments:

    1. a class or factory function that instantiates the desired resource.
    2. `*args` and `**kwargs` that should be supplied to the instantiator.

    .. code-block:: python
        :caption: The function and arguments for creating a file are
            wrapped in a LazyProxy. It is only instantiated when `f.write`
            is called.

        f = LazyProxy(open, "test.txt", mode="r")
        f.write("Hello World!")
        f.close()

    In addition to the class/factory function, it is possible to specifiy a
    `Finaliser`_ supplied to :class:`weakref.finalize` that is called to
    cleanup the resource when the LazyProxy is garbage collected.
    In this case, the first argument should be a tuple of two elements:
    the factory and the finaliser.

    .. code-block:: python

        # LazyProxy defined with factory function and finaliser function
        def finalise_file(file):
            file.close()

        f2 = LazyProxy((open, finalise_file), "test.txt", mode="r")

        class WrappedFile:
            def __init__(self, *args, **kwargs):
                self.handle = open(*args, **kwargs)

            def close(self):
                self.handle.close()

        # LazyProxy defined with class
        f1 = LazyProxy((WrappedFile, WrappedFile.close), "test.txt", mode="r")

    LazyProxy objects are designed to be embedded in
    :func:`dask.array.blockwise` calls.
    For example:

    .. code-block:: python

        # Specify the start and length of each range
        file_ranges = np.array([[0, 5], [5, 10], [15, 5] [20, 10]])
        # Chunk each range individually
        da_file_ranges = dask.array(file_ranges, chunks=(1, 2))
        # Reference a binary file
        file_proxy = LazyProxy(open, "data.dat", "rb")

        def _read(file_proxy, file_range):
            # Seek to range start and read the length of data
            start, length = file_range
            file_proxy.seek(start)
            return np.asarray(file_proxy.read(length), np.uint8)

        data = da.blockwise(_read, "x",
                            # Embed the file_proxy in the graph
                            file_proxy, None,
                            # Pass each file range to the _read
                            da_file_ranges, "xy",
                            # output chunks should have the length
                            # of each range
                            adjust_chunks={"x": tuple(file_ranges[:, 1])},
                            concatenate=True)

        print(data.compute(processes=True))



    Parameters
    ----------
    fn : class or callable or tuple
        A callable object that used to create the proxied object.
        In tuple form, this should consist of two callables.
        The first should create the proxied object and the second
        should be a finaliser that performs cleanup on the proxied object
        when the LazyProxy is garbage collected: it is passed directly
        to :class:`weakref.finalize`.
    *args : tuple
        Positional arguments passed to the callable object
        specified in `fn` that will create the proxied object.
        The contents of `*args` should be pickleable.
    **kwargs : dict
        Keyword arguments passed to the callable object
        specified in `fn` that will create the proxied object.
        The contents of `**kwargs` should be pickleable.

    Notes
    -----
    - Instantiation of the proxied object is thread-safe.
    - LazyProxy's are configured to never instantiate within
      :func:`dask.array.blockwise` and
      :func:`dask.blockwise.blockwise` calls.

    .. _Finaliser: https://en.wikipedia.org/wiki/Finalizer
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
    def __lazy_raise_on_invalid_frames__(cls, frame, depth=10):
        """
        Check that we're not trying to create the proxied object
        in function that inadvertently create it via the use
        of duck-typing.

        Should only be called from :meth:`LazyProxy.__lazy_obj_from_args__`
        and as such, should never be called again once the
        proxied object is created.

        Parameters
        ----------
        frame : :class:`inspect.FrameInfo`
            The calling frame

        depth : int
            Number of frames to search

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
    def __lazy_obj_from_args__(cls, proxy):
        # Double-checked locking
        # https://en.wikipedia.org/wiki/Double-checked_locking

        # getattr_static returns a descriptor for __slots__
        descriptor = getattr_static(proxy, "__lazy_object__")

        try:
            # __lazy_object__ may exist, attempt to return it
            return descriptor.__get__(proxy)
        except AttributeError:
            pass

        # Acquire the creation lock
        with proxy.__lazy_lock__:
            # getattr_static returns a descriptor for __slots__
            descriptor = getattr_static(proxy, "__lazy_object__")

            try:
                # __lazy_object__ may have been created prior to
                # lock acquisition, attempt to return it again
                return descriptor.__get__(proxy)
            except AttributeError:
                # Raise exceptions if we're in an invalid call frame
                cls.__lazy_raise_on_invalid_frames__(inspect.currentframe())

                # Create __lazy_object__
                lazy_obj = proxy.__lazy_fn__(*proxy.__lazy_args__,
                                             **proxy.__lazy_kwargs__)
                proxy.__lazy_object__ = lazy_obj

                # Create finaliser if provided
                if proxy.__lazy_finaliser__:
                    weakref.finalize(proxy,
                                     proxy.__lazy_finaliser__,
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
    """Combination of a :class:`LazyProxy` with a :class:`Multiton`

    Ensures that only a single :class:`LazyProxy` is ever created
    for the given constructor arguments.

    .. code-block:: python

        class A:
            def __init__(self, value):
                self.value = value

        assert LazyProxyMultiton("foo") is LazyProxyMultiton("foo")

    See :class:`LazyProxy` and :class:`Multiton` for further details
    """
