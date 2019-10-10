#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""


import pytest


def test_corr_shape():
    from africanus.util.shapes import corr_shape

    for i in range(10):
        assert corr_shape(i, 'flat') == (i,)

    assert corr_shape(1, 'matrix') == (1,)
    assert corr_shape(2, 'matrix') == (2,)
    assert corr_shape(4, 'matrix') == (2, 2,)

    with pytest.raises(ValueError, match=r"ncorr not in \(1, 2, 4\)"):
        corr_shape(3, 'matrix')


def test_aggregate_chunks():
    from africanus.util.shapes import aggregate_chunks

    chunks, max_c = (3, 4, 6, 3, 6, 7), 10
    expected = (7, 9, 6, 7)
    assert aggregate_chunks(chunks, max_c) == expected

    chunks, max_c = ((3, 4, 6, 3, 6, 7), (1, 1, 1, 1, 1, 1)), (10, 3)
    expected = ((7, 9, 6, 7), (2, 2, 1, 1))
    assert aggregate_chunks(chunks, max_c) == expected

    chunks, max_c = ((3, 4, 6, 3, 6, 7), (1, 1, 1, 1, 1, 1)), (10, 1)
    assert aggregate_chunks(chunks, max_c) == chunks

    chunks, max_c = (5, 5, 5), 5
    assert aggregate_chunks(chunks, max_c) == chunks
