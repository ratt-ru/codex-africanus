"""Keep this file in sync with the dask-ms version"""

import pickle

import pytest

from africanus.util.patterns import Multiton, LazyProxy, LazyProxyMultiton


class DummyResource:
    def __init__(self, arg, tracker, kw=None):
        self.arg = arg
        self.tracker = tracker
        self.kw = kw
        self.value = None

    def set(self, value):
        self.value = value

    def close(self):
        self.tracker.closed = True


class Tracker:
    def __init__(self):
        self.closed = False

    def __eq__(self, other):
        return True

    def __hash__(self):
        return 0


@pytest.mark.parametrize("finalise", [True, False])
@pytest.mark.parametrize("cls", [LazyProxyMultiton, LazyProxy])
def test_lazy(cls, finalise):
    def _inner(tracker):
        if finalise:
            fn = (DummyResource, DummyResource.close)
        else:
            fn = DummyResource

        obj = cls(fn, "test.txt", tracker, kw="w")
        obj.set(5)

        assert obj.arg == "test.txt"
        assert obj.kw == "w"
        assert obj.value == 5

        obj2 = pickle.loads(pickle.dumps(obj))
        assert obj.__lazy_eq__(obj2)
        assert obj.__lazy_hash__() == obj2.__lazy_hash__()

        if cls is LazyProxyMultiton:
            assert obj is obj2
        else:
            assert obj is not obj2

    tracker = Tracker()
    assert tracker.closed is False
    _inner(tracker)
    assert tracker.closed is finalise


def test_multiton():
    class A(metaclass=Multiton):
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class B(metaclass=Multiton):
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    a1 = A(1, 2, 3)
    a2 = A(1, 2)
    a3 = A(1)
    b1 = B(1)

    assert a1 is A(1, 2, 3)
    assert a2 is A(1, 2)
    assert a3 is A(1)
    assert b1 is B(1)
    assert a1 is not a2
    assert a1 is not a3
    assert a1 is not b1
