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

Implements W-Stacking.

.. currentmodule:: africanus.gridding.wstack

.. autosummary::
    w_stacking_layers

.. autofunction:: w_stacking_layers

