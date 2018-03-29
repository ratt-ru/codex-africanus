# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


def on_rtd():
    return bool(os.environ.get("READTHEDOCS"))
