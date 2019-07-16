-----------------------
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

Nifty
~~~~~

Dask wrappers around
`Nifty's Gridder <https://gitlab.mpcdf.mpg.de/ift/nifty_gridder>`_.

Dask
++++

.. currentmodule:: africanus.gridding.nifty.dask

.. autosummary::
    grid_config
    grid
    dirty
    degrid
    model

.. autofunction:: grid_config
.. autofunction:: grid
.. autofunction:: dirty
.. autofunction:: degrid
.. autofunction:: model



W Stacking
~~~~~~~~~~

**This is currently experimental**

Implements W-Stacking as described in `WSClean <wsclean_>`_.

.. currentmodule:: africanus.gridding.wstack

.. autosummary::
    w_stacking_layers
    w_stacking_bins
    w_stacking_centroids
    grid
    degrid

.. autofunction:: w_stacking_layers
.. autofunction:: w_stacking_bins
.. autofunction:: w_stacking_centroids
.. autofunction:: grid
.. autofunction:: degrid

.. _wsclean: https://academic.oup.com/mnras/article/444/1/606/1010067

Utilities
~~~~~~~~~

.. currentmodule:: africanus.gridding.util

.. autosummary::
    estimate_cell_size

.. autofunction:: estimate_cell_size
