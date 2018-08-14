.. _convolution-filter-api:

Convolution Filters
-------------------

Convolution filters suitable for use in gridding and degridding.
Currently, only the Kaiser Bessel filter,
a good approximation of the Prolate Spheroidals, is provided.

.. _kaiser-bessel-filter:

Kaiser Bessel
~~~~~~~~~~~~~

See https://www.dsprelated.com/freebooks/sasp/Kaiser_Window.html.


API
~~~

.. currentmodule:: africanus.filters

.. autosummary::
    convolution_filter


.. autofunction:: convolution_filter
.. autodata:: ConvolutionFilter

