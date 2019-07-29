# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import wraps

from africanus.util.requirements import requires_optional

try:
    from dask.array.core import blockwise
except ImportError as e:
    dask_import_error = e
else:
    dask_import_error = None

DIAG_DIAG = 0
DIAG = 1
FULL = 2
