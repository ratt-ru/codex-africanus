from africanus.rime.fused.transformers.lm import LMTransformer


def test_transformers():
    T = LMTransformer()  # noqa
    assert T.ARGS == ("radec", "phase_centre")
    assert T.KWARGS == {}
    assert T.ALL_ARGS == ("radec", "phase_centre")
