# -*- coding: utf-8 -*-

"""Top-level package for Codex Africanus."""


# NOTE(sjperkins)
# Imports at this level within this module should be avoided,
# or should fail gracefully as this is the base africanus module.
# The setup.py file accesses the ``africanus.install`` modules
import africanus.util.jax_init  # noqa

__author__ = """Simon Perkins"""
__email__ = 'sperkins@ska.ac.za'
__version__ = '0.2.6'
