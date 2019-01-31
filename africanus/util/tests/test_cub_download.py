# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest


def test_cub_download():
    from africanus.util.cub import cub_dir

    cub_dir()
