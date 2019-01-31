# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest


def test_trove_download():
    from africanus.util.trove import trove_dir

    trove_dir()
