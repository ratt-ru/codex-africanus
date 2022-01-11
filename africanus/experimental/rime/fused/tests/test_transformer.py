from africanus.experimental.rime.fused.transformers.lm import LMTransformer


def test_transformers():
    T = LMTransformer()  # noqa
    assert T.ARGS == ("radec", "phase_dir")
    assert T.KWARGS == {}
    assert T.ALL_ARGS == ("radec", "phase_dir")
