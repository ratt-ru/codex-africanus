API
===

Radio Interferometer Measurement Equation
-----------------------------------------

Numpy
~~~~~

.. autofunction:: africanus.rime.phase_delay
.. autofunction:: africanus.rime.brightness
.. autofunction:: africanus.rime.transform_sources
.. autofunction:: africanus.rime.beam_cube_dde

Dask
~~~~

.. autofunction:: africanus.rime.dask.phase_delay
.. autofunction:: africanus.rime.dask.brightness
.. autofunction:: africanus.rime.dask.transform_sources
.. autofunction:: africanus.rime.dask.beam_cube_dde


Simple Gridding
---------------

Numpy
~~~~~

.. autofunction:: africanus.gridding.simple.grid
.. autofunction:: africanus.gridding.simple.degrid

Dask
~~~~~

.. autofunction:: africanus.gridding.simple.dask.grid
.. autofunction:: africanus.gridding.simple.dask.degrid


Convolution Filters
-------------------

.. autodata:: africanus.filters.ConvolutionFilter


.. autofunction:: africanus.filters.convolution_filter

Deconvolution Algorithms
------------------------

.. autofunction:: africanus.deconv.hogbom.hogbom_clean
