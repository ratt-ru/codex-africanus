from africanus.util.patterns import Multiton


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
