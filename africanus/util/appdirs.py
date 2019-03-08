# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join as pjoin

from appdirs import AppDirs

from africanus import __version__

_dirs = AppDirs("codex-africanus", "radio-astronomer", __version__)

user_data_dir = _dirs.user_data_dir
downloads_dir = pjoin(user_data_dir, "downloads")
include_dir = pjoin(user_data_dir, "include")

del __version__
del _dirs
