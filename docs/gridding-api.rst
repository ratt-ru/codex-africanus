-----------------------
Gridding and Degridding
-----------------------

This section contains routines for

1. Gridding complex visibilities onto an image.
2. Degridding complex visibilities from an image.

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

wgridder
~~~~~~~~

Wrappers around 'ducc.wgridder <https://gitlab.mpcdf.mpg.de/mtr/ducc>`_.


Numpy
+++++

.. currentmodule:: africanus.gridding.wgridder

.. autosummary::
    dirty
    model
    residual
    hessian

.. autofunction:: dirty
.. autofunction:: model
.. autofunction:: residual
.. autofunction:: hessian

Dask
++++

.. currentmodule:: africanus.gridding.wgridder.dask

.. autosummary::
    dirty
    model
    residual
    hessian

.. autofunction:: dirty
.. autofunction:: model
.. autofunction:: residual
.. autofunction:: hessian

Utilities
~~~~~~~~~

.. currentmodule:: africanus.gridding.util

.. autosummary::
    estimate_cell_size

.. autofunction:: estimate_cell_size
