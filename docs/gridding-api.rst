Gridding and Degridding
-----------------------

This section contains routines for

1. Gridding complex visibilities onto an image.
2. Degridding complex visibilities from an image.

Simple
~~~~~~

Gridding with no correction for the W-term.

Numpy
+++++

.. currentmodule:: africanus.gridding.simple

.. autosummary::
    grid
    degrid

.. autofunction:: grid
.. autofunction:: degrid


Dask
++++

.. currentmodule:: africanus.gridding.simple.dask

.. autosummary::
    grid
    degrid

.. autofunction:: grid
.. autofunction:: degrid

W Stacking
~~~~~~~~~~

Implements W-Stacking as described in `WSClean <wsclean_>`_.

.. currentmodule:: africanus.gridding.wstack

.. autosummary::
    w_stacking_bins

.. autofunction:: w_stacking_bins

.. _wsclean: https://academic.oup.com/mnras/article/444/1/606/1010067

Utilities
~~~~~~~~~

.. currentmodule:: africanus.gridding.util

.. autosummary::
    cell_size
    uv_scale

.. autofunction:: cell_size
.. autofunction:: uv_scale
