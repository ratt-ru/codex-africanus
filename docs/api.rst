API
===

Radio Interferometer Measurement Equation
-----------------------------------------

Numpy
~~~~~

.. autofunction:: africanus.rime.phase_delay
.. autofunction:: africanus.rime.brightness
.. autofunction:: africanus.rime.parallactic_angles
.. autofunction:: africanus.rime.feed_rotation
.. autofunction:: africanus.rime.transform_sources
.. autofunction:: africanus.rime.beam_cube_dde
.. autofunction:: africanus.rime.zernicke

Dask
~~~~

.. autofunction:: africanus.rime.dask.phase_delay
.. autofunction:: africanus.rime.dask.brightness
.. autofunction:: africanus.rime.dask.parallactic_angles
.. autofunction:: africanus.rime.dask.feed_rotation
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

Utilities
---------

.. autofunction:: africanus.util.beams.beam_filenames
.. autofunction:: africanus.util.beams.beam_grids
