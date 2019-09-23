# -*- coding: utf-8 -*-

"""
Configure jax to use default 64 bit precision
https://github.com/google/jax/issues/216
"""


try:
    import jax.config
except ImportError:
    pass
else:
    jax.config.update("jax_enable_x64", True)
