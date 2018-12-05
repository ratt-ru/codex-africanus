#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from africanus.util.type_inference import infer_complex_dtype


def test_type_inference():
    i32 = np.empty(64, dtype=np.int32)
    i64 = np.empty(64, dtype=np.int64)
    f32 = np.empty(64, dtype=np.float32)
    f64 = np.empty(64, dtype=np.float64)

    assert infer_complex_dtype(f32, f64) == np.complex128
    assert infer_complex_dtype(f32, f32) == np.complex64
    assert infer_complex_dtype(i32, f32) == np.complex128
    assert infer_complex_dtype(i64, f32) == np.complex128
