def test_examples():
    """
    Test that import works at least as this should flush out
    other import issues caused by renames etc.
    """

    from africanus.rime.examples.predict import predict  # noqa

    # TODO(sjperkins)
    # Call with a fake MS once pytest-ms is available
